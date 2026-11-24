
import torch

from src.shie.decoder import CTCDecoder
from src.shie.evaluator import SHIEEvaluator


class SHIETrainer:
    """
    Training and validation engine for SUTRAYA SHIE.

    Responsibilities:
    1. Train the model for one epoch.
    2. Validate the model for one epoch.
    3. Calculate and aggregate training metrics.
    4. Decode predictions and calculate CER/WER.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        gradient_clip=5.0,
        decoder=None
    ):

        if gradient_clip <= 0:
            raise ValueError(
                "gradient_clip must be greater than zero."
            )

        self.device = torch.device(device)

        self.model = model.to(self.device)

        self.criterion = criterion.to(self.device)

        self.optimizer = optimizer

        self.gradient_clip = gradient_clip

        self.decoder = (
            decoder if decoder is not None
            else CTCDecoder()
        )

        self.evaluator = SHIEEvaluator()

    # --------------------------------------------------
    # Prepare a batch
    # --------------------------------------------------

    def _prepare_batch(self, batch):

        required_keys = {
            "images",
            "targets",
            "input_lengths",
            "target_lengths",
            "texts",
            "sample_ids"
        }

        if not isinstance(batch, dict):
            raise TypeError(
                "Every DataLoader batch must be a dictionary."
            )

        missing_keys = required_keys - set(batch.keys())

        if missing_keys:
            raise ValueError(
                f"Missing batch keys: {missing_keys}"
            )

        images = batch["images"].to(
            self.device,
            non_blocking=(self.device.type == "cuda")
        )

        # Keep targets and sequence lengths on CPU.
        # SHIECTCLoss handles the required conversions.
        targets = batch["targets"]

        input_lengths = batch["input_lengths"]

        target_lengths = batch["target_lengths"]

        texts = batch["texts"]

        sample_ids = batch["sample_ids"]

        return (
            images,
            targets,
            input_lengths,
            target_lengths,
            texts,
            sample_ids
        )

    # --------------------------------------------------
    # Training one epoch
    # --------------------------------------------------

    def train_one_epoch(
        self,
        train_loader,
        max_batches=None
    ):

        self.model.train()

        total_loss = 0.0
        total_gradient_norm = 0.0

        total_samples = 0
        total_batches = 0

        for batch_index, batch in enumerate(train_loader):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            (
                images,
                targets,
                input_lengths,
                target_lengths,
                texts,
                sample_ids
            ) = self._prepare_batch(batch)

            batch_size = images.shape[0]

            # 1. Clear previous gradients.
            self.optimizer.zero_grad(
                set_to_none=True
            )

            # 2. Forward propagation.
            logits = self.model(
                images,
                input_lengths=input_lengths
            )

            # 3. Calculate CTC loss.
            loss = self.criterion(
                logits,
                targets,
                input_lengths,
                target_lengths
            )

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite training loss. "
                    f"Samples: {sample_ids}"
                )

            # 4. Backpropagation.
            loss.backward()

            # 5. Gradient clipping.
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip,
                error_if_nonfinite=True
            )

            # 6. Update model parameters.
            self.optimizer.step()

            # 7. Aggregate metrics.
            total_loss += loss.item() * batch_size

            total_gradient_norm += gradient_norm.item()

            total_samples += batch_size

            total_batches += 1

        if total_samples == 0:
            raise RuntimeError(
                "Training DataLoader produced no samples."
            )

        return {
            "loss": total_loss / total_samples,
            "gradient_norm": (
                total_gradient_norm / total_batches
            ),
            "samples": total_samples,
            "batches": total_batches
        }

    # --------------------------------------------------
    # Validation one epoch
    # --------------------------------------------------

    def validate_one_epoch(
        self,
        validation_loader,
        max_batches=None
    ):

        self.model.eval()

        total_loss = 0.0

        total_samples = 0
        total_batches = 0

        references = []
        predictions = []

        # Disable gradient calculation during validation.
        with torch.inference_mode():

            for batch_index, batch in enumerate(
                validation_loader
            ):

                if (
                    max_batches is not None
                    and batch_index >= max_batches
                ):
                    break

                (
                    images,
                    targets,
                    input_lengths,
                    target_lengths,
                    texts,
                    sample_ids
                ) = self._prepare_batch(batch)

                batch_size = images.shape[0]

                # 1. Forward propagation.
                logits = self.model(
                    images,
                    input_lengths=input_lengths
                )

                # 2. Validation CTC loss.
                loss = self.criterion(
                    logits,
                    targets,
                    input_lengths,
                    target_lengths
                )

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        f"Non-finite validation loss. "
                        f"Samples: {sample_ids}"
                    )

                total_loss += loss.item() * batch_size

                # 3. Decode predictions.
                batch_predictions = self.decoder.decode_batch(
                    logits,
                    input_lengths=input_lengths
                )

                references.extend(texts)

                predictions.extend(batch_predictions)

                total_samples += batch_size

                total_batches += 1

        if total_samples == 0:
            raise RuntimeError(
                "Validation DataLoader produced no samples."
            )

        # 4. Calculate aggregate character error rate.
        cer_metrics = self.evaluator.evaluate_cer_dataset(
            references,
            predictions
        )

        # 5. Calculate aggregate word error rate.
        wer_metrics = self.evaluator.evaluate_wer_dataset(
            references,
            predictions
        )

        return {
            "loss": total_loss / total_samples,
            "cer": cer_metrics["error_rate"],
            "wer": wer_metrics["error_rate"],
            "cer_statistics": cer_metrics,
            "wer_statistics": wer_metrics,
            "samples": total_samples,
            "batches": total_batches
        }
