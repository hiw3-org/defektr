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

# Build all sudo calls and batch them into a single block
calls = []
for name, address in TARGETS.items():
    inner = substrate.compose_call(
        call_module="Balances",
        call_function="force_set_balance",
        call_params={"who": address, "new_free": AMOUNT},
    )
    calls.append(substrate.compose_call(
        call_module="Sudo",
        call_function="sudo",
        call_params={"call": inner},
    ))

batch = substrate.compose_call(
    call_module="Utility",
    call_function="batch",
    call_params={"calls": calls},
)

extrinsic = substrate.create_signed_extrinsic(call=batch, keypair=alice)
receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)

if receipt.is_success:
    for name in TARGETS:
        print(f"{name}: ✅")
else:
    print(f"❌ {receipt.error_message}")
