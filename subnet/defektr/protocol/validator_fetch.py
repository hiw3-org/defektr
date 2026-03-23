"""
subnet/defektr/protocol/validator_fetch.py

Reads miner commitments from the chain and downloads model/metadata from IPFS.
Used by the validator's forward loop.
"""

import json
import sys
from pathlib import Path

import bittensor as bt

from defektr.config import NETUID, CACHE_DIR

# Allow importing data.ipfs when this module is used from the project root.
_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # …/defektr/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def fetch_all_cids(subtensor: bt.Subtensor, metagraph: bt.metagraph, netuid: int = NETUID) -> dict[int, str]:
    """
    Read the current metadata CID for every registered miner UID.

    Returns:
        {uid: cid_string}  — empty string if no commit yet.
    """
    cids = {}
    for uid in range(metagraph.n.item()):
        cid = subtensor.get_commitment(netuid, uid) or ""
        cids[uid] = cid
    return cids


def download_metadata(cid: str, uid: int, cache_dir: str = CACHE_DIR) -> dict:
    """
    Download a miner's metadata.json from IPFS by CID.

    Caches to  <cache_dir>/uid_<uid>/metadata.json.
    Returns the parsed metadata dict.
    """
    from data import ipfs  # lazy import — keeps module importable without PINATA_JWT

    dest = Path(cache_dir) / f"uid_{uid}" / "metadata.json"
    dest.parent.mkdir(parents=True, exist_ok=True)

    bt.logging.debug(f"[uid {uid}] Downloading metadata CID {cid} …")
    ipfs.download(cid, str(dest))

    return json.loads(dest.read_text())


def download_model(model_cid: str, uid: int, cache_dir: str = CACHE_DIR) -> str:
    """
    Download a miner's model.onnx from IPFS by the model CID found in metadata.

    Caches to  <cache_dir>/uid_<uid>/model.onnx.
    Returns the local file path.
    """
    from data import ipfs

    dest = Path(cache_dir) / f"uid_{uid}" / "model.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)

    bt.logging.debug(f"[uid {uid}] Downloading model CID {model_cid} …")
    ipfs.download(model_cid, str(dest))

    return str(dest)


def fetch_and_cache_model(
    subtensor: bt.Subtensor,
    uid: int,
    known_cid: str,
    cache: dict,
    cache_dir: str = CACHE_DIR,
    netuid: int = NETUID,
) -> tuple[str | None, bool]:
    """
    High-level helper for the validator forward loop.

    Checks if the miner's metadata CID has changed since last round.
    If so, downloads new metadata + model and updates `cache` in-place.

    Args:
        subtensor:  Connected subtensor.
        uid:        Miner UID.
        known_cid:  Last-seen metadata CID for this UID (empty string = never seen).
        cache:      Mutable dict keyed by uid, storing {"meta_cid", "model_path", "sha256"}.
        cache_dir:  Local directory for downloaded files.
        netuid:     Subnet UID.

    Returns:
        (model_path, updated) — model_path is None if miner hasn't committed yet.
    """
    current_cid = subtensor.get_commitment(netuid, uid) or ""

    if not current_cid:
        return None, False

    if current_cid == known_cid and uid in cache:
        return cache[uid]["model_path"], False  # nothing new

    # New or changed CID — download metadata then model
    bt.logging.info(f"[uid {uid}] New metadata CID {current_cid} — downloading …")
    try:
        metadata   = download_metadata(current_cid, uid, cache_dir)
        model_cid  = metadata["model"]["cid"]
        model_path = download_model(model_cid, uid, cache_dir)

        cache[uid] = {
            "meta_cid":   current_cid,
            "model_path": model_path,
            "sha256":     metadata["model"]["sha256"],
            "metadata":   metadata,
        }
        return model_path, True

    except Exception as e:
        bt.logging.error(f"[uid {uid}] Failed to download model: {e}")
        return None, False
