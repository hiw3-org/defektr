"""
scripts/publish_challenge.py

Validator tool: upload a challenge spec to IPFS and commit the CID on-chain
so miners can discover it via subtensor.get_commitment(netuid, validator_uid).

Usage:
    python scripts/publish_challenge.py \
        --spec   challenge_spec.json \
        --wallet-name validator --wallet-hotkey hotkey_0 \
        --wallet-path /home/luka/ws/bittensor_test/wallets \
        --network ws://127.0.0.1:9944

    # Update block numbers automatically from the current block:
    python scripts/publish_challenge.py --spec challenge_spec.json --update-blocks
"""

import sys
import json
import argparse
from pathlib import Path
from copy import deepcopy

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import bittensor as bt
from data import ipfs
from defektr.config import NETUID, CHALLENGE_INTERVAL


def publish(
    spec_path: str,
    wallet: bt.Wallet,
    subtensor: bt.Subtensor,
    update_blocks: bool = False,
    netuid: int = NETUID,
) -> str:
    """
    Upload challenge_spec.json to IPFS and commit the CID from the validator wallet.

    Returns the metadata CID string.
    """
    spec = json.loads(Path(spec_path).read_text())

    if update_blocks:
        current = subtensor.get_current_block()
        # Align challenge_block to the next CHALLENGE_INTERVAL boundary
        challenge_block = ((current // CHALLENGE_INTERVAL) + 1) * CHALLENGE_INTERVAL
        deadline_block  = challenge_block + CHALLENGE_INTERVAL
        spec = deepcopy(spec)
        spec["challenge_block"] = challenge_block
        spec["deadline_block"]  = deadline_block
        print(f"Updated challenge_block={challenge_block}  deadline_block={deadline_block}")

    # Write updated spec to a temp file for upload
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                    prefix="defektr_spec_") as f:
        json.dump(spec, f, indent=2)
        tmp_path = f.name

    try:
        print(f"Uploading {spec_path} to IPFS …")
        result = ipfs.upload(
            tmp_path,
            name=f"{spec['challenge_id']}_challenge_spec.json",
            keyvalues={
                "challenge_id":  spec["challenge_id"],
                "type":          "challenge_spec",
                "validator":     wallet.hotkey.ss58_address,
            },
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    cid = result["cid"]
    print(f"Challenge spec CID: {cid}")

    print(f"Committing CID on-chain (netuid={netuid}) …")
    subtensor.set_commitment(wallet, netuid, cid)
    print(f"✅ Done. Miners can fetch spec via:")
    print(f"   subtensor.get_commitment({netuid}, <validator_uid>)  → {cid}")
    return cid


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Publish a Defektr challenge spec to IPFS and on-chain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spec",            default="challenge_spec.json")
    p.add_argument("--wallet-name",     default="validator")
    p.add_argument("--wallet-hotkey",   default="hotkey_0")
    p.add_argument("--wallet-path",     default="/home/luka/ws/bittensor_test/wallets")
    p.add_argument("--network",         default="ws://127.0.0.1:9944")
    p.add_argument("--netuid",          type=int, default=NETUID)
    p.add_argument("--update-blocks",   action="store_true",
                   help="Auto-update challenge_block and deadline_block from current block.")
    args = p.parse_args()

    wallet    = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey, path=args.wallet_path)
    subtensor = bt.Subtensor(network=args.network)

    publish(
        spec_path     = args.spec,
        wallet        = wallet,
        subtensor     = subtensor,
        update_blocks = args.update_blocks,
        netuid        = args.netuid,
    )
