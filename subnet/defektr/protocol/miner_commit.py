"""
subnet/defektr/protocol/miner_commit.py

Thin wrapper around subtensor.commit() for announcing a metadata CID on-chain.
Used by training/upload.py (and any future automation).
"""

import bittensor as bt
from defektr.config import NETUID


def commit_cid(wallet: bt.Wallet, subtensor: bt.Subtensor, cid: str, netuid: int = NETUID) -> None:
    """
    Announce a Pinata IPFS metadata CID on-chain.

    Args:
        wallet:   Miner wallet (signs the extrinsic).
        subtensor: Connected subtensor instance.
        cid:      The IPFS CID of the metadata.json file.
        netuid:   Subnet UID (defaults to NETUID from config).
    """
    bt.logging.info(f"Committing CID {cid!r} to netuid {netuid} …")
    subtensor.commit(wallet, netuid, cid)
    bt.logging.info("Commit successful.")


def get_committed_cid(subtensor: bt.Subtensor, uid: int, netuid: int = NETUID) -> str:
    """
    Read the latest committed metadata CID for a miner UID.

    Returns an empty string if no commit has been made yet.
    """
    cid = subtensor.get_commitment(netuid, uid) or ""
    return cid
