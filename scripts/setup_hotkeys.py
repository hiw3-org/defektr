"""
scripts/setup_hotkeys.py

One-shot setup for the Defektr hackathon submission:
  - Creates 10 miner hotkeys  (miner wallet, hotkeys hotkey_0 … hotkey_9)
  - Creates  3 validator hotkeys (validator wallet, hotkeys hotkey_0 … hotkey_2)
  - Funds all coldkeys via Alice sudo (top-up to 2000 TAO each)
  - Registers all 13 hotkeys on netuid 3
  - Adds 100 TAO stake to each validator hotkey

Run AFTER the chain is up and subnet is created:
    python defektr/scripts/setup_hotkeys.py

Or from the project root (bittensor_test/):
    python defektr/scripts/setup_hotkeys.py

Re-run safely after a Docker restart — hotkeys already on disk are skipped,
registration is idempotent (burned_register will no-op if already registered).
"""

import sys
import time
from pathlib import Path

import bittensor as bt

# ── Config ────────────────────────────────────────────────────────────────────

ENDPOINT    = "ws://127.0.0.1:9944"
WALLET_PATH = str(Path(__file__).resolve().parent.parent.parent / "wallets")
NETUID      = 3
FUND_RAO    = 2_000 * 10**9   # 2000 TAO per coldkey
STAKE_TAO   = 100             # TAO to stake per validator hotkey

MINER_WALLET     = "miner"
VALIDATOR_WALLET = "validator"
N_MINERS         = 10
N_VALIDATORS     = 3

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fund_coldkey(substrate, address: str, label: str):
    alice = bt.Keypair.create_from_uri("//Alice")
    call = substrate.compose_call(
        call_module="Balances",
        call_function="force_set_balance",
        call_params={"who": address, "new_free": FUND_RAO},
    )
    sudo_call = substrate.compose_call(
        call_module="Sudo",
        call_function="sudo",
        call_params={"call": call},
    )
    extrinsic = substrate.create_signed_extrinsic(call=sudo_call, keypair=alice)
    receipt   = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
    status = "✅" if receipt.is_success else f"❌ {receipt.error_message}"
    print(f"  Fund {label}: {status}")


def _register(subtensor, wallet, label: str):
    if subtensor.is_hotkey_registered(netuid=NETUID, hotkey_ss58=wallet.hotkey.ss58_address):
        print(f"  Register {label}: already registered ✅")
        return
    result = subtensor.burned_register(wallet=wallet, netuid=NETUID)
    print(f"  Register {label}: {'✅' if result else '❌'}")


def _add_stake(subtensor, wallet, label: str):
    try:
        subtensor.add_stake(
            wallet        = wallet,
            hotkey_ss58   = wallet.hotkey.ss58_address,
            netuid        = NETUID,
            amount        = bt.Balance.from_tao(STAKE_TAO),
        )
        print(f"  Stake {label}: ✅ +{STAKE_TAO} TAO")
    except Exception as e:
        print(f"  Stake {label}: ❌ {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Connecting to {ENDPOINT} …")
    subtensor = bt.Subtensor(network=ENDPOINT)
    substrate = subtensor.substrate

    # ── Step 1: Create hotkeys on disk ───────────────────────────────────────
    print(f"\n[1/4] Creating hotkeys …  (wallet path: {WALLET_PATH})")

    miner_wallets     = []
    validator_wallets = []

    for i in range(N_MINERS):
        hotkey_name = f"hotkey_{i}"
        w = bt.Wallet(name=MINER_WALLET, hotkey=hotkey_name, path=WALLET_PATH)
        if not w.hotkey_file.exists_on_device():
            w.create_new_hotkey(use_password=False, overwrite=False)
            print(f"  Created miner/{hotkey_name}  {w.hotkey.ss58_address}")
        else:
            print(f"  Exists  miner/{hotkey_name}  {w.hotkey.ss58_address}")
        miner_wallets.append(w)

    for i in range(N_VALIDATORS):
        hotkey_name = f"hotkey_{i}"
        w = bt.Wallet(name=VALIDATOR_WALLET, hotkey=hotkey_name, path=WALLET_PATH)
        if not w.hotkey_file.exists_on_device():
            w.create_new_hotkey(use_password=False, overwrite=False)
            print(f"  Created validator/{hotkey_name}  {w.hotkey.ss58_address}")
        else:
            print(f"  Exists  validator/{hotkey_name}  {w.hotkey.ss58_address}")
        validator_wallets.append(w)

    # ── Step 2: Fund coldkeys ─────────────────────────────────────────────────
    # We only need to fund each unique coldkey once (all hotkeys share it).
    print(f"\n[2/4] Funding coldkeys ({FUND_RAO // 10**9} TAO each) …")
    miner_coldkey = bt.Wallet(name=MINER_WALLET, path=WALLET_PATH).coldkey.ss58_address
    val_coldkey   = bt.Wallet(name=VALIDATOR_WALLET, path=WALLET_PATH).coldkey.ss58_address
    _fund_coldkey(substrate, miner_coldkey, f"miner coldkey ({miner_coldkey[:8]}…)")
    _fund_coldkey(substrate, val_coldkey,   f"validator coldkey ({val_coldkey[:8]}…)")

    time.sleep(2)  # Let balances settle

    # ── Step 3: Register all hotkeys ─────────────────────────────────────────
    print(f"\n[3/4] Registering {N_MINERS} miner + {N_VALIDATORS} validator hotkeys on netuid {NETUID} …")
    for w in miner_wallets:
        _register(subtensor, w, f"miner/{w.hotkey_str}")
        time.sleep(1)   # avoid tx flood

    for w in validator_wallets:
        _register(subtensor, w, f"validator/{w.hotkey_str}")
        time.sleep(1)

    # ── Step 4: Stake validators ──────────────────────────────────────────────
    print(f"\n[4/4] Adding {STAKE_TAO} TAO stake to each validator hotkey …")
    for w in validator_wallets:
        _add_stake(subtensor, w, f"validator/{w.hotkey_str}")
        time.sleep(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n─── Summary ───────────────────────────────────────────")
    metagraph = subtensor.metagraph(NETUID)
    for w in miner_wallets + validator_wallets:
        hotkey = w.hotkey.ss58_address
        if hotkey in metagraph.hotkeys:
            uid = metagraph.hotkeys.index(hotkey)
            stake = float(metagraph.S[uid])
            print(f"  {w.name}/{w.hotkey_str:12s}  uid={uid:3d}  stake={stake:.1f} τ")
        else:
            print(f"  {w.name}/{w.hotkey_str:12s}  NOT REGISTERED ❌")

    print("\nDone. Run the full reset after a Docker restart:")
    print("  python defektr/scripts/setup_hotkeys.py")


if __name__ == "__main__":
    main()
