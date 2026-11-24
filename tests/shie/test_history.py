
import json
import shutil
from pathlib import Path

from src.shie.history import SHIEHistoryManager


def run_history_tests():

    print("\n========== SHIE HISTORY TESTS ==========\n")

    test_dir = Path("tests/temp_shie_history")

    if test_dir.exists():
        shutil.rmtree(test_dir)

    history_path = test_dir / "training_history.json"

    manager = SHIEHistoryManager(history_path)

    # -----------------------------------------
    # TEST 1: Initialize empty history
    # -----------------------------------------

    history = manager.get_history()

    assert history["project"] == "SUTRAYA"
    assert history["module"] == "SHIE"
    assert history["epochs"] == []

    print("TEST 1: History initialization PASSED")

    # -----------------------------------------
    # TEST 2: Add first epoch
    # -----------------------------------------

    training_metrics = {
        "loss": 20.45,
        "gradient_norm": 4.82,
        "samples": 6482,
        "batches": 811
    }

    validation_metrics = {
        "loss": 15.32,
        "cer": 0.95,
        "wer": 0.98
    }

    manager.add_epoch(
        epoch=1,
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        learning_rate=0.001
    )

    assert len(manager.get_history()["epochs"]) == 1

    print("TEST 2: Add epoch PASSED")

    # -----------------------------------------
    # TEST 3: Add another epoch
    # -----------------------------------------

    manager.add_epoch(
        epoch=2,
        training_metrics={
            "loss": 15.20,
            "gradient_norm": 3.12,
            "samples": 6482,
            "batches": 811
        },
        validation_metrics={
            "loss": 12.10,
            "cer": 0.80,
            "wer": 0.91
        },
        learning_rate=0.001
    )

    assert len(manager.get_history()["epochs"]) == 2
    assert manager.get_latest_epoch() == 2

    print("TEST 3: Multiple epochs PASSED")

    # -----------------------------------------
    # TEST 4: Prevent duplicate epoch
    # -----------------------------------------

    manager.add_epoch(
        epoch=2,
        training_metrics={"loss": 10.0},
        validation_metrics={"cer": 0.70, "wer": 0.80},
        learning_rate=0.0005
    )

    epochs = manager.get_history()["epochs"]

    assert len(epochs) == 2
    assert epochs[-1]["training"]["loss"] == 10.0

    print("TEST 4: Duplicate epoch handling PASSED")

    # -----------------------------------------
    # TEST 5: Check chronological ordering
    # -----------------------------------------

    manager.add_epoch(
        epoch=4,
        training_metrics={"loss": 8.0},
        validation_metrics={"cer": 0.60, "wer": 0.70},
        learning_rate=0.0005
    )

    manager.add_epoch(
        epoch=3,
        training_metrics={"loss": 9.0},
        validation_metrics={"cer": 0.65, "wer": 0.75},
        learning_rate=0.0005
    )

    epochs = manager.get_history()["epochs"]

    assert [
        item["epoch"] for item in epochs
    ] == [1, 2, 3, 4]

    print("TEST 5: Chronological ordering PASSED")

    # -----------------------------------------
    # TEST 6: Load saved history
    # -----------------------------------------

    new_manager = SHIEHistoryManager(history_path)

    loaded_epochs = new_manager.get_history()["epochs"]

    assert len(loaded_epochs) == 4
    assert new_manager.get_latest_epoch() == 4

    print("TEST 6: History restoration PASSED")

    # -----------------------------------------
    # TEST 7: Validate JSON file
    # -----------------------------------------

    with history_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        saved_data = json.load(file)

    assert saved_data["project"] == "SUTRAYA"
    assert len(saved_data["epochs"]) == 4

    print("TEST 7: JSON file validation PASSED")

    # -----------------------------------------
    # CLEANUP
    # -----------------------------------------

    shutil.rmtree(test_dir)

    print("\n========================================")
    print("ALL SHIE HISTORY TESTS PASSED!")
    print("========================================")


if __name__ == "__main__":
    run_history_tests()
