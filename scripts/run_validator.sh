#!/bin/bash
# Run validator and save logs to logs/validator_<hotkey>_<timestamp>.log

HOTKEY=${1:-default}
NETWORK=${2:-wss://test.finney.opentensor.ai}
WALLETS=${3:-/home/luka/ws/bittensor_test/defektr/wallets}

mkdir -p "$(dirname "$0")/../logs"
LOGFILE="$(dirname "$0")/../logs/validator_${HOTKEY}_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOGFILE"

python "$(dirname "$0")/../subnet/neurons/validator.py" \
    --netuid 29 \
    --subtensor.network "$NETWORK" \
    --wallet.name validator \
    --wallet.hotkey "$HOTKEY" \
    --wallet.path "$WALLETS" \
    --defektr.spec "$(dirname "$0")/../challenge_spec.json" \
    --defektr.val_dataset "$(dirname "$0")/../data/datasets/bottle" \
    --logging.debug \
    2>&1 | tee "$LOGFILE"
