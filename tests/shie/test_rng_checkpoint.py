
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn

from src.shie.checkpoint import SHIECheckpointManager


def test_rng_checkpoint_restoration():

    print("\n========== RNG CHECKPOINT TEST ==========\n")

    # Temporary checkpoint directory
    with tempfile.TemporaryDirectory() as temp_dir:

        manager = SHIECheckpointManager(temp_dir)

        # Small model for checkpoint testing
        model = nn.Linear(4, 2)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min"
        )

        # --------------------------------------------------
        # 1. INITIALIZE RANDOM GENERATORS
        # --------------------------------------------------

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        print("TEST 1: Random generators initialized")

        # --------------------------------------------------
        # 2. SAVE CHECKPOINT
        # --------------------------------------------------

        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            best_cer=0.8,
            training_metrics={"loss": 10.0},
            validation_metrics={
                "loss": 12.0,
                "cer": 0.8,
                "wer": 0.9
            },
            scheduler=scheduler
        )

        print("TEST 2: Checkpoint saved")

        # --------------------------------------------------
        # 3. GENERATE EXPECTED RANDOM VALUES
        # --------------------------------------------------

        expected_python = random.random()

        expected_numpy = np.random.random()

        expected_torch = torch.rand(3)

        expected_cuda = None

        if torch.cuda.is_available():

            expected_cuda = torch.rand(
                3,
                device="cuda"
            ).cpu()

        print("TEST 3: Expected values generated")

        # --------------------------------------------------
        # 4. CHANGE RANDOM STATES
        # --------------------------------------------------

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(999)

        print("TEST 4: Random states changed")

        # --------------------------------------------------
        # 5. LOAD CHECKPOINT
        # --------------------------------------------------

        metadata = manager.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        assert metadata["epoch"] == 1

        print("TEST 5: Checkpoint loaded")

        # --------------------------------------------------
        # 6. GENERATE ACTUAL RANDOM VALUES
        # --------------------------------------------------

        actual_python = random.random()

        actual_numpy = np.random.random()

        actual_torch = torch.rand(3)

        actual_cuda = None

        if torch.cuda.is_available():

            actual_cuda = torch.rand(
                3,
                device="cuda"
            ).cpu()

        # --------------------------------------------------
        # 7. COMPARE RANDOM VALUES
        # --------------------------------------------------

        assert expected_python == actual_python

        assert expected_numpy == actual_numpy

        assert torch.equal(
            expected_torch,
            actual_torch
        )

        if torch.cuda.is_available():

            assert torch.equal(
                expected_cuda,
                actual_cuda
            )

        print("TEST 6: Python RNG restoration PASSED")
        print("TEST 7: NumPy RNG restoration PASSED")
        print("TEST 8: PyTorch CPU RNG restoration PASSED")

        if torch.cuda.is_available():
            print("TEST 9: PyTorch CUDA RNG restoration PASSED")

        print("\nALL RNG CHECKPOINT TESTS PASSED!")


if __name__ == "__main__":
    test_rng_checkpoint_restoration()
