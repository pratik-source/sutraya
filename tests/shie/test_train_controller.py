
import copy
import json
import shutil
from pathlib import Path

import torch

from src.shie.config import load_config
from src.shie.train import train_shie


def run_controller_tests():

    print("\n========== SHIE CONTROLLER INTEGRATION TEST ==========\n")

    test_dir = Path("tests/temp_shie_controller")

    if test_dir.exists():
        shutil.rmtree(test_dir)

    checkpoint_dir = test_dir / "checkpoints"
    history_path = test_dir / "training_history.json"

    try:

        # ---------------------------------------------
        # TEST 1: Load and prepare test configuration
        # ---------------------------------------------

        config = copy.deepcopy(load_config())

        config["training"]["epochs"] = 1
        config["checkpoint"]["directory"] = str(checkpoint_dir)

        print("\nTEST 1: Configuration prepared")

        # ---------------------------------------------
        # TEST 2: Run one controlled training epoch
        # ---------------------------------------------

        history = train_shie(
            config=config,
            resume=False,
            max_train_batches=1,
            max_validation_batches=1,
            history_path=str(history_path)
        )

        print("\nTEST 2: Controller execution completed")

        # ---------------------------------------------
        # TEST 3: Verify training history
        # ---------------------------------------------

        assert history_path.exists(), (
            "Training history file was not created."
        )

        with history_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            saved_history = json.load(file)

        assert len(saved_history["epochs"]) == 1

        epoch_record = saved_history["epochs"][0]

        assert epoch_record["epoch"] == 1
        assert epoch_record["training"]["batches"] == 1
        assert epoch_record["validation"]["batches"] == 1

        assert epoch_record["training"]["samples"] > 0
        assert epoch_record["validation"]["samples"] > 0

        assert epoch_record["learning_rate"] > 0

        print("TEST 3: Persistent history PASSED")

        # ---------------------------------------------
        # TEST 4: Verify latest checkpoint
        # ---------------------------------------------

        last_checkpoint_path = (
            checkpoint_dir / "last_checkpoint.pth"
        )

        assert last_checkpoint_path.exists()

        checkpoint = torch.load(
            last_checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        assert checkpoint["epoch"] == 1
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint

        print("TEST 4: Latest checkpoint PASSED")

        # ---------------------------------------------
        # TEST 5: Verify best model
        # ---------------------------------------------

        best_model_path = (
            checkpoint_dir / "best_model.pth"
        )

        # The first validation CER should normally be
        # available for this non-empty validation batch.

        assert best_model_path.exists(), (
            "Best model checkpoint was not created."
        )

        best_model = torch.load(
            best_model_path,
            map_location="cpu",
            weights_only=True
        )

        assert best_model["epoch"] == 1
        assert "model_state_dict" in best_model
        assert "validation_cer" in best_model

        print("TEST 5: Best model checkpoint PASSED")

        # ---------------------------------------------
        # TEST 6: Verify metrics
        # ---------------------------------------------

        assert epoch_record["training"]["loss"] >= 0
        assert epoch_record["validation"]["loss"] >= 0

        assert 0 <= epoch_record["validation"]["cer"]
        assert 0 <= epoch_record["validation"]["wer"]

        print("TEST 6: Training and validation metrics PASSED")

        print("\n==============================================")
        print("ALL SHIE CONTROLLER INTEGRATION TESTS PASSED!")
        print("==============================================")

    finally:

        if test_dir.exists():
            shutil.rmtree(test_dir)

        print("\nTemporary test files cleaned up.")


if __name__ == "__main__":
    run_controller_tests()
