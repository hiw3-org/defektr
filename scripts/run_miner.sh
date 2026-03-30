#!/bin/bash
# Run miner and save logs to logs/miner_<wallet>_<hotkey>_<timestamp>.log

WALLET_NAME=${1:-miner}
HOTKEY=${2:-hotkey_0}
NETWORK=${3:-wss://test.finney.opentensor.ai}
WALLETS=${4:-/home/luka/ws/bittensor_test/defektr/wallets}

mkdir -p "$(dirname "$0")/../logs"
LOGFILE="$(dirname "$0")/../logs/miner_${WALLET_NAME}_${HOTKEY}_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOGFILE"

python "$(dirname "$0")/../subnet/neurons/miner.py" \
    --netuid 29 \
    --subtensor.network "$NETWORK" \
    --wallet.name "$WALLET_NAME" \
    --wallet.hotkey "$HOTKEY" \
    --wallet.path "$WALLETS" \
    --logging.debug \
    2>&1 | tee "$LOGFILE"
