import bittensor as bt

ENDPOINT = "ws://127.0.0.1:9944"
WALLET_PATH = "wallets"
NETUID = 2

subtensor = bt.Subtensor(network=ENDPOINT)

for name in ["miner", "validator"]:
    wallet = bt.Wallet(name=name, hotkey="default", path=WALLET_PATH)
    result = subtensor.burned_register(wallet=wallet, netuid=NETUID)
    print(f"{name}: {'✅' if result else '❌'}")
