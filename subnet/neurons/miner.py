"""
subnet/neurons/miner.py

Defektr miner process.

Defektr miners do NOT serve an API endpoint — they upload trained ONNX models
to IPFS and announce the metadata CID on-chain via subtensor.commit().
This process keeps the miner registered and active on the network, and
watches for new challenge specs from the validator.

Uploading a model is done separately via:
    python training/upload.py --model <path> --spec challenge_spec.json ...

Run:
    python subnet/neurons/miner.py \
        --netuid 3 \
        --subtensor.network ws://127.0.0.1:9944 \
        --wallet.name miner \
        --wallet.hotkey hotkey_0 \
        --wallet.path /home/luka/ws/bittensor_test/wallets \
        --logging.debug
"""

import sys
import time
import argparse
from pathlib import Path

import bittensor as bt

_SUBNET_DIR = Path(__file__).resolve().parent.parent
if str(_SUBNET_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBNET_DIR))

from defektr.config import NETUID, CHALLENGE_INTERVAL
from defektr.protocol.miner_commit import get_committed_cid


def get_config() -> bt.Config:
    parser = argparse.ArgumentParser(description="Defektr miner")
    parser.add_argument("--netuid",  type=int, default=NETUID)
    parser.add_argument("--subtensor.network", type=str, default="ws://127.0.0.1:9944",
                        dest="subtensor_network")
    parser.add_argument("--wallet.name",    type=str, default="miner",   dest="wallet_name")
    parser.add_argument("--wallet.hotkey",  type=str, default="default",  dest="wallet_hotkey")
    parser.add_argument("--wallet.path",    type=str, default="wallets",  dest="wallet_path")
    parser.add_argument("--logging.debug",  action="store_true",          dest="logging_debug")
    parser.add_argument("--poll_interval",  type=int, default=60,
                        help="Seconds between chain polls.")
    args = parser.parse_args()

    cfg = bt.Config()
    cfg.netuid              = args.netuid
    cfg.subtensor_network   = args.subtensor_network
    cfg.wallet_name         = args.wallet_name
    cfg.wallet_hotkey       = args.wallet_hotkey
    cfg.wallet_path         = args.wallet_path
    cfg.logging_debug       = args.logging_debug
    cfg.poll_interval       = args.poll_interval
    return cfg


def main():
    cfg = get_config()

    if cfg.logging_debug:
        bt.logging.set_debug(True)

    wallet = bt.Wallet(
        name   = cfg.wallet_name,
        hotkey = cfg.wallet_hotkey,
        path   = cfg.wallet_path,
    )
    subtensor = bt.Subtensor(network=cfg.subtensor_network)
    metagraph = subtensor.metagraph(cfg.netuid)

    bt.logging.info(f"Defektr miner starting")
    bt.logging.info(f"  Hotkey  : {wallet.hotkey.ss58_address}")
    bt.logging.info(f"  Network : {cfg.subtensor_network}")
    bt.logging.info(f"  Netuid  : {cfg.netuid}")

    # Check registration
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"Hotkey {wallet.hotkey.ss58_address} is not registered on netuid {cfg.netuid}. "
            f"Run register_neurons.py first."
        )
        return

    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"  UID     : {my_uid}")

    last_challenge_block = None

    try:
        while True:
            block     = subtensor.get_current_block()
            chal_blk  = (block // CHALLENGE_INTERVAL) * CHALLENGE_INTERVAL

            # Log committed CID
            committed_cid = get_committed_cid(subtensor, my_uid, cfg.netuid)

            if chal_blk != last_challenge_block:
                bt.logging.info(
                    f"[block {block}] New challenge period started at block {chal_blk}"
                )
                last_challenge_block = chal_blk

            if committed_cid:
                bt.logging.info(
                    f"[block {block}] uid={my_uid} committed_cid={committed_cid[:20]}…"
                )
            else:
                bt.logging.warning(
                    f"[block {block}] uid={my_uid} no model committed yet. "
                    f"Run: python training/upload.py --model <path> --spec challenge_spec.json ..."
                )

            # Resync metagraph periodically
            metagraph.sync(subtensor=subtensor)

            time.sleep(cfg.poll_interval)

    except KeyboardInterrupt:
        bt.logging.success("Miner stopped by keyboard interrupt.")


if __name__ == "__main__":
    main()
