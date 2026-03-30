import bittensor as bt
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "subnet"))
from defektr.config import NETUID

s = bt.Subtensor(network="wss://test.finney.opentensor.ai")
m = s.metagraph(NETUID)
print(f"{'UID':<5} {'hotkey':<50} {'incentive':<12} {'stake':<12}")
print("-" * 80)
for uid in range(len(m.hotkeys)):
    print(f"{uid:<5} {m.hotkeys[uid]:<50} {float(m.incentive[uid]):<12.4f} {float(m.total_stake[uid]):<12.4f}")
