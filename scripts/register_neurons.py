import logging
import bittensor as bt
logging.getLogger("bittensor").setLevel(logging.CRITICAL)

ENDPOINT = "ws://127.0.0.1:9944"
WALLET_PATH = "/home/luka/ws/bittensor_test/defektr/wallets"
NETUID = 2

subtensor = bt.Subtensor(network=ENDPOINT)

wallet = bt.Wallet(name="validator", hotkey="default", path=WALLET_PATH)
result = subtensor.burned_register(wallet=wallet, netuid=NETUID)
print(f"validator: {'✅' if result else '❌'}")
