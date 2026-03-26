"""
subnet/neurons/validator.py

Defektr validator neuron.

Run:
    python subnet/neurons/validator.py \
        --netuid 3 \
        --subtensor.network ws://127.0.0.1:9944 \
        --wallet.name validator \
        --wallet.hotkey default \
        --wallet.path /home/luka/ws/bittensor_test/wallets \
        --defektr.spec challenge_spec.json \
        --defektr.val_dataset data/datasets/validation_dataset \
        --logging.debug
"""

import sys
import time
import argparse
from pathlib import Path

import logging
import bittensor as bt
import bittensor.utils.networking as _bt_net

# We don't use Dendrite/Axon — skip the external-IP lookup that requires internet.
_bt_net.get_external_ip = lambda: "127.0.0.1"

# Suppress the SDK's "NoneType not subscriptable" spam from get_commitment()
# calls on UIDs that have never committed (e.g. subnet owner uid=0).
_orig_bt_error = bt.logging.error
def _filtered_bt_error(msg, *args, **kwargs):
    if "NoneType" not in str(msg):
        _orig_bt_error(msg, *args, **kwargs)
bt.logging.error = _filtered_bt_error

# Make sure the subnet package is importable when run from project root.
_SUBNET_DIR = Path(__file__).resolve().parent.parent
if str(_SUBNET_DIR) not in sys.path:
    sys.path.insert(0, str(_SUBNET_DIR))

from template.base.validator import BaseValidatorNeuron
from defektr.validator.forward import forward


class Validator(BaseValidatorNeuron):
    """
    Defektr validator.  Inherits all boilerplate (metagraph sync, EMA scoring,
    set_weights) from BaseValidatorNeuron.  The only custom logic is forward().
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument(
            "--defektr.spec",
            type=str,
            default="challenge_spec.json",
            help="Path to challenge_spec.json (loaded each forward pass).",
        )
        parser.add_argument(
            "--defektr.val_dataset",
            type=str,
            default="data/datasets/validation_dataset",
            help="Path to the generated validation dataset root.",
        )

    def __init__(self, config=None):
        super().__init__(config=config)

        # In-memory model cache: {uid: {"meta_cid", "model_path", "sha256", "metadata"}}
        self.model_cache: dict = {}

        # Flatten dotted config keys for easy access in forward.py
        self.config.defektr_spec        = self.config.defektr.spec
        self.config.defektr_val_dataset = self.config.defektr.val_dataset

        bt.logging.info("Defektr validator initialised.")
        bt.logging.info(f"  Challenge spec : {self.config.defektr_spec}")
        bt.logging.info(f"  Val dataset    : {self.config.defektr_val_dataset}")

        self.load_state()

    async def concurrent_forward(self):
        # Run exactly one forward at a time — our substrate calls are synchronous
        # and the WebSocket can't handle concurrent recv() from multiple coroutines.
        await self.forward()

    async def forward(self):
        return await forward(self)


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            try:
                bt.logging.info(
                    f"Validator running | block={validator.block} "
                    f"step={validator.step} "
                    f"cached_models={len(validator.model_cache)}"
                )
            except Exception:
                pass
            time.sleep(12)
