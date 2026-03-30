import logging
import bittensor as bt
logging.getLogger("bittensor").setLevel(logging.CRITICAL)

ENDPOINT = "wss://test.finney.opentensor.ai"
WALLET_PATH = "/home/luka/ws/bittensor_test/defektr/wallets" # Update this to your wallets directory if different
NETUID = 2

subtensor = bt.Subtensor(network=ENDPOINT)

wallet = bt.Wallet(name="validator", hotkey="default", path=WALLET_PATH)
result = subtensor.burned_register(wallet=wallet, netuid=NETUID)
print(f"validator: {'OK' if result else 'ERROR'}")
