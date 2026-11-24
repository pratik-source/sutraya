
import copy
import json
import shutil
from pathlib import Path

import torch

from src.shie.config import load_config
from src.shie.train import train_shie


def run_resume_tests():

    print("\n========== SHIE RESUME TRAINING TEST ==========\n")

    test_dir = Path("tests/temp_shie_resume")

    if test_dir.exists():
        shutil.rmtree(test_dir)

    checkpoint_dir = test_dir / "checkpoints"
    history_path = test_dir / "training_history.json"

    try:

        config = copy.deepcopy(load_config())

        config["training"]["epochs"] = 1
        config["checkpoint"]["directory"] = str(checkpoint_dir)

        # -----------------------------------------
        # TEST 1: First execution - Epoch 1
        # -----------------------------------------

        print("\nTEST 1: Starting first training session")

        train_shie(
            config=config,
            resume=False,
            max_train_batches=1,
            max_validation_batches=1,
            history_path=str(history_path)
        )

        last_checkpoint = checkpoint_dir / "last_checkpoint.pth"

        assert last_checkpoint.exists()

        checkpoint = torch.load(
            last_checkpoint,
            map_location="cpu",
            weights_only=True
        )

        assert checkpoint["epoch"] == 1

        print("TEST 1: Epoch 1 checkpoint saved")

        # -----------------------------------------
        # TEST 2: Simulate interruption
        # -----------------------------------------

        print("\nTEST 2: First session ended")

        # The first train_shie() call has returned.
        # Its model, optimizer and scheduler objects
        # are no longer used by the second execution.

        # -----------------------------------------
        # TEST 3: Resume from Epoch 1
        # -----------------------------------------

        config["training"]["epochs"] = 2

        print("\nTEST 3: Resuming training")

        resumed_history = train_shie(
            config=config,
            resume=True,
            max_train_batches=1,
            max_validation_batches=1,
            history_path=str(history_path)
        )

        print("TEST 3: Resume execution completed")

        # -----------------------------------------
        # TEST 4: Verify final checkpoint
        # -----------------------------------------

        checkpoint = torch.load(
            last_checkpoint,
            map_location="cpu",
            weights_only=True
        )

        assert checkpoint["epoch"] == 2

        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint

        print("TEST 4: Epoch 2 checkpoint verified")

        # -----------------------------------------
        # TEST 5: Verify persistent history
        # -----------------------------------------

        with history_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            saved_history = json.load(file)

        epochs = saved_history["epochs"]

        assert len(epochs) == 2

        assert epochs[0]["epoch"] == 1
        assert epochs[1]["epoch"] == 2

        assert resumed_history["epochs"] == epochs

        print("TEST 5: Both epochs preserved")

        # -----------------------------------------
        # TEST 6: Verify best model
        # -----------------------------------------

        best_model_path = checkpoint_dir / "best_model.pth"

        assert best_model_path.exists()

        best_model = torch.load(
            best_model_path,
            map_location="cpu",
            weights_only=True
        )

        assert best_model["epoch"] in (1, 2)

        print("TEST 6: Best model preserved")

        print("\n==========================================")
        print("ALL SHIE RESUME TESTS PASSED!")
        print("==========================================")

    finally:

        if test_dir.exists():
            shutil.rmtree(test_dir)

        print("\nTemporary test files cleaned up.")


if __name__ == "__main__":
    run_resume_tests()
