import bittensor as bt

ENDPOINT = "ws://127.0.0.1:9944"
AMOUNT = 2_000 * 10**9  # 2000 TAO in rao

TARGETS = {
    "owner":     "5DqCkgJsmyGHDHzGtFMgJP2VzjSxpJa2L4F6PwoFuATZypBc",
    "validator": "5HbNykjW5YL6y2dK5hYRakpDHCQ8WTDk42mejEpBHdLSRjzN",
    "miner":     "5E573Qv4D9JW6UniETaQ3wtX1LXT8UwG7dWMQ3EPxaszw5oS",
}

alice = bt.Keypair.create_from_uri("//Alice")
print(f"Alice: {alice.ss58_address}")

substrate = bt.Subtensor(network=ENDPOINT).substrate

for name, address in TARGETS.items():
    call = substrate.compose_call(
        call_module="Balances",
        call_function="force_set_balance",
        call_params={"who": address, "new_free": AMOUNT},
    )
    sudo_call = substrate.compose_call(
        call_module="Sudo",
        call_function="sudo",
        call_params={"call": call},
    )
    extrinsic = substrate.create_signed_extrinsic(call=sudo_call, keypair=alice)
    receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
    status = "✅" if receipt.is_success else f"❌ {receipt.error_message}"
    print(f"{name}: {status}")
