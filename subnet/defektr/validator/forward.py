"""
subnet/defektr/validator/forward.py

Core validator forward pass for Defektr.

Each epoch the validator:
  1. Reads every miner's on-chain metadata CID commitment.
  2. Downloads new/changed models to the local cache.
  3. Runs each cached model against a seed-stable benchmark batch.
  4. Scores with reward() and updates the EMA score table.

No Axon / Synapse / dendrite is used — miners are evaluated purely from
their uploaded ONNX files.
"""

import sys
import json
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import bittensor as bt

# Project root on sys.path so we can import data.*
_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # …/defektr/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from defektr.config import NETUID, CHALLENGE_INTERVAL, CACHE_DIR, BENCHMARK_N
from defektr.validator.reward import reward as compute_reward
from data.sampler import sample_benchmark, load_image_as_array


# ──────────────────────────────────────────────── Constants ───────────────────

_DEFAULT_SPEC_PATH   = str(_ROOT / "challenge_spec.json")
_THRESHOLD           = 0.5     # sigmoid threshold for hard binary prediction
_HARDWARE_FACTOR     = 1.0     # set to 15.0 for real edge simulation in scoring
                               # (upload.py uses 15x for the gate; here we measure
                               #  raw validator-hardware latency for the speed reward)


# ──────────────────────────────────────────────── Inference helpers ───────────

def _run_model(
    session: ort.InferenceSession,
    records: list,
    input_name: str,
    n_warmup: int = 2,
) -> tuple[list, float]:
    """
    Run ``session`` on every image in ``records`` and collect predictions.

    Returns:
        predictions  — list of {"image_id", "confidence", "mask"} dicts
        mean_latency_s — mean per-image wall-clock time (seconds)
    """
    predictions = []
    latencies   = []

    # Warmup
    if records:
        dummy = load_image_as_array(records[0]["image_path"])
        for _ in range(n_warmup):
            session.run(None, {input_name: dummy})

    for rec in records:
        img = load_image_as_array(rec["image_path"])  # [1, 3, H, W]

        t0      = time.perf_counter()
        outputs = session.run(None, {input_name: img})
        latencies.append(time.perf_counter() - t0)

        confidence = float(outputs[0].flatten()[0])

        pred_mask = None
        if len(outputs) > 1:
            # Segmentation head: [1, 1, H, W] → [H, W] binary
            raw_mask  = outputs[1].squeeze()           # [H, W] float32
            pred_mask = (raw_mask >= _THRESHOLD).astype(np.uint8)

        predictions.append({
            "image_id":  rec["image_id"],
            "confidence": confidence,
            "mask":       pred_mask,
        })

    mean_latency_s = float(np.mean(latencies)) if latencies else 0.0
    return predictions, mean_latency_s


def _score_uid(
    uid: int,
    model_path: str,
    metadata: dict,
    benchmark: list,
    challenge_spec: dict,
) -> float:
    """
    Load the cached ONNX model for ``uid``, run it against the benchmark
    batch, and return a composite reward in [0, 1].

    Returns 0.0 on any error (model crash, shape mismatch, etc.).
    """
    try:
        input_name = metadata.get("inference", {}).get("input_name", "image")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        predictions, mean_latency_s = _run_model(session, benchmark, input_name)

        score = compute_reward(
            ground_truth   = benchmark,
            predictions    = predictions,
            mean_latency_s = mean_latency_s * _HARDWARE_FACTOR,
            challenge_spec = challenge_spec,
        )

        bt.logging.info(
            f"[uid {uid}] score={score:.4f}  latency={mean_latency_s*1000:.1f}ms"
            f"  outputs={'2 (seg)' if any(p['mask'] is not None for p in predictions) else '1 (bin)'}"
        )
        return score

    except Exception as e:
        bt.logging.error(f"[uid {uid}] inference failed: {e}")
        bt.logging.debug(traceback.format_exc())
        return 0.0


# ──────────────────────────────────────────────── Main forward ────────────────

async def forward(self) -> None:
    """
    Validator forward pass.  Called once per epoch by BaseValidatorNeuron.run().

    ``self`` is the Validator instance — provides:
        self.subtensor, self.metagraph, self.wallet,
        self.block, self.model_cache,
        self.update_scores(), self.config
    """
    # ── Load challenge spec ──────────────────────────────────────────────────
    spec_path = getattr(self.config, "defektr_spec", _DEFAULT_SPEC_PATH)
    try:
        challenge_spec = json.loads(Path(spec_path).read_text())
    except Exception as e:
        bt.logging.error(f"Cannot load challenge spec from {spec_path}: {e}")
        return

    netuid = self.config.netuid

    # ── Identify miner UIDs (all registered, excluding self) ─────────────────
    try:
        my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
    except ValueError:
        my_uid = -1

    miner_uids = [
        uid for uid in range(self.metagraph.n.item())
        if uid != my_uid
    ]

    if not miner_uids:
        bt.logging.warning("No miner UIDs found.")
        return

    bt.logging.info(f"Scoring {len(miner_uids)} miners at block {self.block}")

    # ── Update model cache for any UIDs with new commits ─────────────────────
    from defektr.protocol.validator_fetch import fetch_and_cache_model

    for uid in miner_uids:
        known_cid = self.model_cache.get(uid, {}).get("meta_cid", "")
        try:
            fetch_and_cache_model(
                subtensor  = self.subtensor,
                uid        = uid,
                known_cid  = known_cid,
                cache      = self.model_cache,
                cache_dir  = CACHE_DIR,
                netuid     = netuid,
            )
        except Exception as e:
            bt.logging.warning(f"[uid {uid}] cache update failed: {e}")

    # ── Sample benchmark images (seed = challenge block hash) ─────────────────
    val_dataset_path = getattr(
        self.config, "defektr_val_dataset",
        str(_ROOT / "data" / "datasets" / "validation_dataset")
    )

    challenge_block = (self.block // CHALLENGE_INTERVAL) * CHALLENGE_INTERVAL
    try:
        block_hash = self.subtensor.get_block_hash(challenge_block)
        seed = block_hash if block_hash else challenge_block
    except Exception:
        seed = challenge_block

    try:
        benchmark = sample_benchmark(val_dataset_path, seed=seed, n=BENCHMARK_N)
        bt.logging.info(f"Benchmark: {len(benchmark)} images (seed={seed!r})")
    except Exception as e:
        bt.logging.error(f"Failed to load benchmark dataset: {e}")
        return

    # ── Score each miner ──────────────────────────────────────────────────────
    scores = np.zeros(len(miner_uids), dtype=np.float32)

    for i, uid in enumerate(miner_uids):
        entry = self.model_cache.get(uid)
        if entry is None:
            bt.logging.debug(f"[uid {uid}] no model cached — score 0")
            continue

        scores[i] = _score_uid(
            uid            = uid,
            model_path     = entry["model_path"],
            metadata       = entry.get("metadata", {}),
            benchmark      = benchmark,
            challenge_spec = challenge_spec,
        )

    # ── Update EMA scores ─────────────────────────────────────────────────────
    self.update_scores(scores, miner_uids)

    ranked = sorted(zip(miner_uids, scores), key=lambda x: x[1], reverse=True)
    bt.logging.info(
        "Scores this epoch: "
        + "  ".join(f"uid={u} {s:.4f}" for u, s in ranked)
    )
