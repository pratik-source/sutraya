
import json
from pathlib import Path


class SHIEHistoryManager:

    def __init__(self, history_path="experiments/shie/training_history.json"):

        self.history_path = Path(history_path)

        self.history_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.history = {
            "project": "SUTRAYA",
            "module": "SHIE",
            "epochs": []
        }

        self.load_history()

    # --------------------------------------------------
    # LOAD EXISTING HISTORY
    # --------------------------------------------------

    def load_history(self):

        if not self.history_path.exists():
            print("No previous training history found.")
            return self.history

        with self.history_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            loaded_history = json.load(file)

        if not isinstance(loaded_history, dict):
            raise ValueError(
                "Training history must be a JSON object."
            )

        if loaded_history.get("project") != "SUTRAYA":
            raise ValueError(
                "Invalid project name in training history."
            )

        if loaded_history.get("module") != "SHIE":
            raise ValueError(
                "Invalid module name in training history."
            )

        epochs = loaded_history.get("epochs")

        if not isinstance(epochs, list):
            raise ValueError(
                "Training history must contain an epochs list."
            )

        self.history = loaded_history

        print(
            f"Training history loaded: {len(epochs)} epochs"
        )

        return self.history

    # --------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------

    def save_history(self):

        temporary_path = self.history_path.with_suffix(".tmp")

        with temporary_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.history,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False
            )

        temporary_path.replace(self.history_path)

        print("Training history saved successfully.")

    # --------------------------------------------------
    # ADD OR UPDATE AN EPOCH
    # --------------------------------------------------

    def add_epoch(
        self,
        epoch,
        training_metrics,
        validation_metrics,
        learning_rate
    ):

        if not isinstance(epoch, int) or epoch < 1:
            raise ValueError(
                "Epoch must be a positive integer."
            )

        if not isinstance(training_metrics, dict):
            raise TypeError(
                "Training metrics must be a dictionary."
            )

        if not isinstance(validation_metrics, dict):
            raise TypeError(
                "Validation metrics must be a dictionary."
            )

        if learning_rate <= 0:
            raise ValueError(
                "Learning rate must be positive."
            )

        epoch_record = {
            "epoch": epoch,
            "training": training_metrics,
            "validation": validation_metrics,
            "learning_rate": learning_rate
        }

        # Remove an existing record for the same epoch.
        # This prevents duplicate records after resuming.

        self.history["epochs"] = [
            item
            for item in self.history["epochs"]
            if item["epoch"] != epoch
        ]

        self.history["epochs"].append(epoch_record)

        # Keep records in chronological order.

        self.history["epochs"].sort(
            key=lambda item: item["epoch"]
        )

        self.save_history()

        return epoch_record

    # --------------------------------------------------
    # GET COMPLETE HISTORY
    # --------------------------------------------------

    def get_history(self):

        return self.history

    # --------------------------------------------------
    # GET LATEST COMPLETED EPOCH
    # --------------------------------------------------

    def get_latest_epoch(self):

        epochs = self.history["epochs"]

        if not epochs:
            return None

        return max(
            item["epoch"]
            for item in epochs
        )
