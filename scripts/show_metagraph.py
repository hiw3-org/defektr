import bittensor as bt

s = bt.Subtensor(network="ws://127.0.0.1:9944")
m = s.metagraph(2)
print(f"{'UID':<5} {'hotkey':<50} {'incentive':<12} {'stake':<12}")
print("-" * 80)
for uid in range(len(m.hotkeys)):
    print(f"{uid:<5} {m.hotkeys[uid]:<50} {float(m.incentive[uid]):<12.4f} {float(m.total_stake[uid]):<12.4f}")
