import logging
import bittensor as bt
logging.getLogger("bittensor").setLevel(logging.CRITICAL)                                                                                                                                                                                                                    
subtensor = bt.Subtensor(network='ws://127.0.0.1:9944')                                                                                                                                                                                   
wallet = bt.Wallet(name='validator', hotkey='default', path='/home/luka/ws/bittensor_test/defektr/wallets')                                                                                                                                       
NETUID = 2
subtensor.add_stake(
	wallet=wallet,
	hotkey_ss58=wallet.hotkey.ss58_address,
	netuid=NETUID,
	amount=bt.Balance.from_tao(100)
)
print('✅ Stake added')
