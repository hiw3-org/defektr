"""
scripts/add_stake_all.py

Add stake to all miner and validator hotkeys via root network (netuid 0).
Staking to netuid 29 directly is disabled (SubtokenDisabled) on testnet,
but staking to root network satisfies the AccountNotAllowedCommit check
and shows up in the metagraph stake column.
"""

import sys
from pathlib import Path
import bittensor as bt

ENDPOINT    = "wss://test.finney.opentensor.ai"
WALLET_PATH = str(Path(__file__).resolve().parent.parent / "wallets")
STAKE_TAO   = 10

WALLETS = [
    # (wallet_name, hotkey_name)
    ("miner",    "hotkey_0"),
    ("miner2",   "default"),
    ("miner3",   "default"),
    ("miner4",   "default"),
    ("miner5",   "default"),
    ("miner6",   "default"),
    ("miner7",   "default"),
    ("miner8",   "default"),
    ("miner9",   "default"),
    ("miner10",  "default"),
    ("validator", "hotkey_0"),
    ("validator", "hotkey_1"),
    ("validator", "hotkey_2"),
]

subtensor = bt.Subtensor(network=ENDPOINT)

for wallet_name, hotkey_name in WALLETS:
    w = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=WALLET_PATH)
    label = f"{wallet_name}/{hotkey_name}"
    try:
        subtensor.add_stake(
            wallet      = w,
            hotkey_ss58 = w.hotkey.ss58_address,
            netuid      = 0,   # root network — avoids SubtokenDisabled
            amount      = bt.Balance.from_tao(STAKE_TAO),
        )
        print(f"  Stake {label}: OK +{STAKE_TAO} TAO")
    except Exception as e:
        print(f"  Stake {label}: ERROR {e}")
