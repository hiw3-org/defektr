"""
training/upload.py — Defektr miner upload tool.

Validates a trained ONNX model against the challenge spec, uploads it
(plus a metadata sidecar) to IPFS, and announces the metadata CID on-chain.

Importable:
    from training.upload import upload_and_commit
    cid = upload_and_commit("model.onnx", challenge_spec, wallet, subtensor)

CLI:
    python training/upload.py \\
        --model  training/models/baseline.onnx \\
        --spec   challenge_spec.json \\
        --wallet-name miner --wallet-hotkey default \\
        --wallet-path /home/luka/ws/bittensor_test/wallets \\
        --network ws://127.0.0.1:9944

Add --dry-run to validate + simulate without uploading or committing.
"""

import sys
import json
import hashlib
import tempfile
import time
from pathlib import Path

# Allow running as `python training/upload.py` from project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import onnx
import onnxruntime as ort


# ──────────────────────────────────────────────── ONNX inspection ─────────────

_DTYPE_ALIASES = {"float": "float32", "double": "float64"}


def inspect_onnx(model_path: str) -> dict:
    """
    Load and inspect an ONNX model.

    Returns:
        {
          "input_name":  str,
          "input_shape": [B, C, H, W],
          "input_dtype": "float32",
          "outputs":     [{"name": str, "shape": [...]}, ...],
          "size_mb":     float,
        }
    """
    path = Path(model_path)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    def _shape(vi):
        return [d.dim_value for d in vi.type.tensor_type.shape.dim]

    inp = model.graph.input[0]
    raw_dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type).lower()

    outputs = []
    for out in model.graph.output:
        s = _shape(out)
        outputs.append({"name": out.name, "shape": s})

    return {
        "input_name":  inp.name,
        "input_shape": _shape(inp),
        "input_dtype": _DTYPE_ALIASES.get(raw_dtype, raw_dtype),
        "outputs":     outputs,
        "size_mb":     round(path.stat().st_size / 1024 / 1024, 3),
    }


# ──────────────────────────────────────────────── Validation ──────────────────

def validate_against_spec(onnx_info: dict, spec: dict) -> None:
    """
    Hard-check model info against challenge spec constraints.
    Raises ValueError on any mismatch.

    Checks: size, input dtype, input spatial shape.
    Output shape is logged as a warning only (dynamic batch dims make exact
    comparison unreliable, and the improved model has two outputs).
    """
    c = spec["constraints"]
    errors = []

    # Size
    if onnx_info["size_mb"] > c["max_model_size_mb"]:
        errors.append(
            f"size {onnx_info['size_mb']} MB > limit {c['max_model_size_mb']} MB"
        )

    # Input dtype
    if onnx_info["input_dtype"] != c["input_dtype"]:
        errors.append(
            f"input dtype {onnx_info['input_dtype']!r} != spec {c['input_dtype']!r}"
        )

    # Input spatial shape — spec is [C, H, W], model should be [B, C, H, W]
    spec_chw = list(c["input_shape"])          # [C, H, W]
    model_shape = onnx_info["input_shape"]     # [B, C, H, W]
    if len(model_shape) != 4:
        errors.append(f"expected 4-D input, got shape {model_shape}")
    elif list(model_shape[1:]) != spec_chw:
        errors.append(
            f"input spatial shape {model_shape[1:]} != spec {spec_chw}"
        )

    if errors:
        raise ValueError("Model failed spec validation:\n  " + "\n  ".join(errors))

    # Soft output check — just inform, don't fail
    if not onnx_info["outputs"]:
        raise ValueError("Model has no outputs.")

    primary = onnx_info["outputs"][0]
    spec_out = list(c.get("output_shape", []))
    if spec_out:
        # Strip dynamic (0) dims for comparison
        static_model = [d for d in primary["shape"] if d != 0]
        static_spec  = [d for d in spec_out          if d != 0]
        if static_model != static_spec:
            print(
                f"  [warn] primary output shape {primary['shape']} "
                f"vs spec {spec_out} (may be OK if batch dim is dynamic)"
            )


# ──────────────────────────────────────────────── Edge deployability ───────────

def check_edge_deployability(
    model_path: str,
    onnx_info: dict,
    spec: dict,
    hardware_factor: float = 15.0,
    n_warmup: int = 3,
    n_timed: int = 20,
) -> dict:
    """
    Run single-threaded CPU inference and apply a hardware_factor penalty
    to simulate edge hardware (e.g. RPi4).

    Returns:
        {"passed": bool, "simulated_fps": float, "simulated_ms": float,
         "measured_ms": float, "hardware_factor": float}

    Raises ValueError if simulated FPS < spec min_fps.
    """
    c = spec["constraints"]
    min_fps = c["min_fps"]
    C, H, W = c["input_shape"]

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    dummy = np.random.rand(1, C, H, W).astype(np.float32)
    feed  = {onnx_info["input_name"]: dummy}

    for _ in range(n_warmup):
        session.run(None, feed)

    t0 = time.perf_counter()
    for _ in range(n_timed):
        session.run(None, feed)
    measured_ms   = (time.perf_counter() - t0) / n_timed * 1000
    simulated_ms  = measured_ms * hardware_factor
    simulated_fps = 1000.0 / simulated_ms

    result = {
        "passed":          simulated_fps >= min_fps,
        "simulated_fps":   round(simulated_fps, 2),
        "simulated_ms":    round(simulated_ms,  1),
        "measured_ms":     round(measured_ms,   1),
        "hardware_factor": hardware_factor,
    }

    if not result["passed"]:
        raise ValueError(
            f"Edge deployability FAIL: {simulated_fps:.1f} FPS simulated "
            f"< {min_fps} FPS required "
            f"({measured_ms:.1f} ms × {hardware_factor}x = {simulated_ms:.1f} ms/image)"
        )

    return result


# ──────────────────────────────────────────────── Helpers ─────────────────────

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(
    challenge_spec: dict,
    onnx_info: dict,
    model_cid: str,
    model_sha256: str,
    miner_hotkey: str,
    submitted_block: int,
    model_filename: str = "model.onnx",
    architecture: str = None,
) -> dict:
    """Build the metadata JSON that gets uploaded alongside the model."""
    c = challenge_spec["constraints"]
    return {
        "challenge_id":   challenge_spec["challenge_id"],
        "miner_hotkey":   miner_hotkey,
        "submitted_block": submitted_block,
        "model": {
            "filename": model_filename,
            "cid":      model_cid,
            "size_mb":  onnx_info["size_mb"],
            "sha256":   model_sha256,
        },
        "inference": {
            "input_name":    onnx_info["input_name"],
            "input_shape":   onnx_info["input_shape"],
            "input_dtype":   onnx_info["input_dtype"],
            "outputs":       onnx_info["outputs"],
            "normalization": c.get("normalization"),
            "threshold":     c.get("threshold", 0.5),
        },
        "training": {
            "architecture": architecture,
            "dataset":      challenge_spec.get("dataset"),
            "category":     challenge_spec.get("category"),
        },
    }


# ──────────────────────────────────────────────── Main pipeline ───────────────

def upload_and_commit(
    model_path: str,
    challenge_spec: dict,
    wallet,
    subtensor,
    architecture: str = None,
    hardware_factor: float = 15.0,
    dry_run: bool = False,
) -> str:
    """
    Full miner upload pipeline:
      1. Inspect ONNX
      2. Validate against challenge spec
      3. Check edge deployability (simulated)
      4. Compute SHA-256
      5. Upload model.onnx to IPFS
      6. Build metadata.json and upload to IPFS
      7. Commit metadata CID on-chain via subtensor.commit()

    Returns the metadata CID string (empty string on dry-run).
    """
    from data import ipfs  # imported here so the module works without bittensor/pinata at import time

    model_path = str(model_path)

    # 0. Merge external data into a single self-contained file if needed
    data_file = Path(model_path).with_suffix("") .parent / (Path(model_path).stem + ".onnx.data")
    if not data_file.exists():
        data_file = Path(str(model_path) + ".data")
    _merged_tmp = None
    if data_file.exists():
        print(f"[0/7] Merging external data file into single ONNX …")
        import onnx as _onnx
        _m = _onnx.load(model_path)
        _tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False, prefix="defektr_merged_")
        _tmp.close()
        _onnx.save(_m, _tmp.name, save_as_external_data=False)
        model_path = _tmp.name
        _merged_tmp = _tmp.name
        print(f"      merged → {model_path} ({Path(model_path).stat().st_size/1024/1024:.1f} MB)")

    # 1. Inspect
    print(f"\n[1/7] Inspecting {model_path} …")
    onnx_info = inspect_onnx(model_path)
    print(f"      input : {onnx_info['input_name']} {onnx_info['input_shape']} {onnx_info['input_dtype']}")
    for out in onnx_info["outputs"]:
        print(f"      output: {out['name']} {out['shape']}")
    print(f"      size  : {onnx_info['size_mb']} MB")

    # 2. Validate against spec
    print("[2/7] Validating against challenge spec …")
    validate_against_spec(onnx_info, challenge_spec)
    print("      ✓ input shape / dtype / size OK")

    # 3. Edge deployability
    print(f"[3/7] Checking edge deployability (hardware_factor={hardware_factor}x) …")
    edge = check_edge_deployability(model_path, onnx_info, challenge_spec, hardware_factor)
    print(
        f"      {edge['measured_ms']} ms measured × {hardware_factor}x"
        f" = {edge['simulated_ms']} ms simulated → {edge['simulated_fps']} FPS ✓"
    )

    # 4. SHA-256
    print("[4/7] Computing SHA-256 …")
    sha256 = sha256_file(model_path)
    print(f"      {sha256}")

    if dry_run:
        if _merged_tmp:
            Path(_merged_tmp).unlink(missing_ok=True)
        print("\n[dry-run] Validation passed. Stopping before upload.")
        return ""

    # 5. Upload model
    print("[5/7] Uploading model to IPFS …")
    model_result = ipfs.upload(
        model_path,
        name=f"{challenge_spec['challenge_id']}_model.onnx",
        keyvalues={
            "challenge_id": challenge_spec["challenge_id"],
            "miner_hotkey": wallet.hotkey.ss58_address,
            "type":         "model",
        },
    )
    model_cid = model_result["cid"]
    print(f"      model CID: {model_cid}")

    # 6. Build + upload metadata
    print("[6/7] Building and uploading metadata …")
    current_block = subtensor.get_current_block()
    metadata = build_metadata(
        challenge_spec  = challenge_spec,
        onnx_info       = onnx_info,
        model_cid       = model_cid,
        model_sha256    = sha256,
        miner_hotkey    = wallet.hotkey.ss58_address,
        submitted_block = current_block,
        model_filename  = Path(model_path).name,
        architecture    = architecture,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="defektr_meta_"
    ) as f:
        json.dump(metadata, f, indent=2)
        meta_tmp = f.name

    try:
        meta_result = ipfs.upload(
            meta_tmp,
            name=f"{challenge_spec['challenge_id']}_metadata.json",
            keyvalues={
                "challenge_id": challenge_spec["challenge_id"],
                "miner_hotkey": wallet.hotkey.ss58_address,
                "type":         "metadata",
            },
        )
    finally:
        Path(meta_tmp).unlink(missing_ok=True)

    meta_cid = meta_result["cid"]
    print(f"      metadata CID: {meta_cid}")

    # 7. On-chain commit
    print("[7/7] Committing metadata CID on-chain …")
    netuid = challenge_spec.get("netuid", 2)
    subtensor.set_commitment(wallet, netuid, meta_cid)
    print(f"      ✓ committed to netuid {netuid}")

    if _merged_tmp:
        Path(_merged_tmp).unlink(missing_ok=True)

    print(f"\nDone. Metadata CID: {meta_cid}")
    return meta_cid


# ──────────────────────────────────────────────── CLI ─────────────────────────

if __name__ == "__main__":
    import argparse
    import bittensor as bt

    p = argparse.ArgumentParser(
        description="Defektr miner upload tool — validate, upload, and commit an ONNX model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",           required=True,  help="Path to model.onnx")
    p.add_argument("--spec",            required=True,  help="Path to challenge_spec.json")
    p.add_argument("--wallet-name",     default="miner")
    p.add_argument("--wallet-hotkey",   default="default")
    p.add_argument("--wallet-path",     default="wallets")
    p.add_argument("--network",         default="ws://127.0.0.1:9944")
    p.add_argument("--architecture",    default=None,   help="Architecture name (stored in metadata)")
    p.add_argument("--hardware-factor", type=float, default=15.0,
                   help="Penalty multiplier simulating edge hardware vs benchmark CPU")
    p.add_argument("--dry-run",         action="store_true",
                   help="Validate only — no IPFS upload or on-chain commit")
    args = p.parse_args()

    challenge_spec = json.loads(Path(args.spec).read_text())

    wallet    = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey, path=args.wallet_path)
    subtensor = bt.Subtensor(network=args.network)

    upload_and_commit(
        model_path     = args.model,
        challenge_spec = challenge_spec,
        wallet         = wallet,
        subtensor      = subtensor,
        architecture   = args.architecture,
        hardware_factor= args.hardware_factor,
        dry_run        = args.dry_run,
    )
