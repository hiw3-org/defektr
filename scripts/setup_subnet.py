import bittensor as bt

ENDPOINT = "ws://127.0.0.1:9944"
WALLET_PATH = "/home/luka/ws/bittensor_test/defektr/wallets"

wallet = bt.Wallet(name="owner", hotkey="default", path=WALLET_PATH)
subtensor = bt.Subtensor(network=ENDPOINT)

print(f"Owner balance: {subtensor.get_balance(wallet.coldkeypub.ss58_address)}")

result = subtensor.register_subnet(wallet=wallet)
print(f"✅ Result: {result}")
