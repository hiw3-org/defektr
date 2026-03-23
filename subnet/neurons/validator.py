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

import bittensor as bt

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

    async def forward(self):
        return await forward(self)


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(
                f"Validator running | block={validator.block} "
                f"step={validator.step} "
                f"cached_models={len(validator.model_cache)}"
            )
            time.sleep(12)
