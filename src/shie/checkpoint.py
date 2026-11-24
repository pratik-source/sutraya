
from pathlib import Path
import random
import warnings

import numpy as np
import torch


class SHIECheckpointManager:

    def __init__(self, checkpoint_dir):

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.last_checkpoint_path = (
            self.checkpoint_dir / "last_checkpoint.pth"
        )

        self.best_model_path = (
            self.checkpoint_dir / "best_model.pth"
        )

    # --------------------------------------------------
    # RNG STATE MANAGEMENT
    # --------------------------------------------------

    @staticmethod
    def _capture_rng_state():
        """
        Capture random-number-generator states from
        Python, NumPy, PyTorch CPU and CUDA.
        """

        numpy_state = np.random.get_state()

        rng_state = {
            "python": random.getstate(),

            "numpy": {
                "bit_generator": numpy_state[0],
                "state": numpy_state[1].tolist(),
                "pos": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4])
            },

            "torch_cpu": torch.get_rng_state(),

            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            )
        }

        return rng_state

    @staticmethod
    def _restore_rng_state(rng_state):
        """
        Restore random-number-generator states.

        Older checkpoints without RNG information are
        handled by load_checkpoint().
        """

        # Restore Python RNG
        random.setstate(
            rng_state["python"]
        )

        # Restore NumPy RNG
        numpy_state = rng_state["numpy"]

        np.random.set_state((
            numpy_state["bit_generator"],
            np.asarray(
                numpy_state["state"],
                dtype=np.uint32
            ),
            numpy_state["pos"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"]
        ))

        # Restore PyTorch CPU RNG
        torch.set_rng_state(
            rng_state["torch_cpu"].cpu()
        )

        # Restore PyTorch CUDA RNG
        cuda_states = rng_state["torch_cuda"]

        if cuda_states is not None:

            if torch.cuda.is_available():

                if len(cuda_states) == torch.cuda.device_count():

                    torch.cuda.set_rng_state_all([
                        state.cpu()
                        for state in cuda_states
                    ])

                else:
                    warnings.warn(
                        "CUDA device count differs from the "
                        "saved checkpoint. CUDA RNG states "
                        "were not restored."
                    )

            else:
                warnings.warn(
                    "Checkpoint contains CUDA RNG states, "
                    "but CUDA is unavailable. CUDA RNG "
                    "states were not restored."
                )

    # --------------------------------------------------
    # SAVE LATEST TRAINING CHECKPOINT
    # --------------------------------------------------

    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch,
        best_cer,
        training_metrics,
        validation_metrics,
        scheduler=None
    ):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_cer": best_cer,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,

            # Reproducibility information
            "rng_state": self._capture_rng_state()
        }

        if scheduler is not None:

            checkpoint["scheduler_state_dict"] = (
                scheduler.state_dict()
            )

        temporary_path = self.last_checkpoint_path.with_suffix(
            ".tmp"
        )

        torch.save(checkpoint, temporary_path)

        temporary_path.replace(self.last_checkpoint_path)

        print(
            f"Checkpoint saved successfully: Epoch {epoch}"
        )

    # --------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------

    def save_best_model(
        self,
        model,
        epoch,
        validation_metrics
    ):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "validation_metrics": validation_metrics,
            "validation_cer": validation_metrics["cer"]
        }

        temporary_path = self.best_model_path.with_suffix(
            ".tmp"
        )

        torch.save(checkpoint, temporary_path)

        temporary_path.replace(self.best_model_path)

        print(
            f"New best SHIE model saved: "
            f"CER = {validation_metrics['cer']:.4f}"
        )

    # --------------------------------------------------
    # LOAD LATEST TRAINING CHECKPOINT
    # --------------------------------------------------

    def load_checkpoint(
        self,
        model,
        optimizer=None,
        scheduler=None,
        device="cpu"
    ):

        if not self.last_checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{self.last_checkpoint_path}"
            )

        checkpoint = torch.load(
            self.last_checkpoint_path,
            map_location=device,
            weights_only=True
        )

        # Restore model
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Restore optimizer
        if optimizer is not None:

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        # Restore scheduler
        if scheduler is not None:

            if "scheduler_state_dict" not in checkpoint:

                raise KeyError(
                    "Scheduler state not found in checkpoint."
                )

            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        # Restore RNG states LAST.
        # This ensures model/optimizer restoration does not
        # alter the random sequence we want to continue from.
        if "rng_state" in checkpoint:

            self._restore_rng_state(
                checkpoint["rng_state"]
            )

            print("Random-number-generator states restored.")

        else:

            warnings.warn(
                "This checkpoint does not contain RNG states. "
                "Training can resume, but exact random-state "
                "continuation is unavailable."
            )

        print(
            f"Checkpoint loaded successfully: "
            f"Epoch {checkpoint['epoch']}"
        )

        return {
            "epoch": checkpoint["epoch"],
            "best_cer": checkpoint["best_cer"],
            "training_metrics": checkpoint["training_metrics"],
            "validation_metrics": checkpoint["validation_metrics"]
        }
