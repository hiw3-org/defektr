import bittensor as bt

ENDPOINT = "wss://test.finney.opentensor.ai"
WALLET_PATH = "/home/luka/ws/bittensor_test/defektr/wallets"

wallet = bt.Wallet(name="owner", hotkey="default", path=WALLET_PATH)
subtensor = bt.Subtensor(network=ENDPOINT)

print(f"Balance: {subtensor.get_balance(wallet.coldkeypub.ss58_address)}")
print(f"Current block: {subtensor.get_current_block()}")

result = subtensor.register_subnet(wallet=wallet)
print(f"Result: {result}")