
import csv
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.shie.config import load_config, resolve_project_path
from src.shie.model import SHIEModel
from src.shie.dataset import IAMDataset, collate_iam_batch
from src.shie.bucket_sampler import WidthBucketBatchSampler
from src.shie.decoder import CTCDecoder
from src.shie.evaluator import SHIEEvaluator
from src.shie.ctc_loss import SHIECTCLoss


def resolve_device(device_config):
    """Resolve the device used for SHIE evaluation."""

    if device_config == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_config == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no GPU is available."
            )

        return torch.device("cuda")

    return torch.device("cpu")


def create_test_dataloader(config):
    """Create a deterministic DataLoader for IAM test samples."""

    dataset_config = config["dataset"]
    training_config = config["training"]
    hardware_config = config["hardware"]

    test_dataset = IAMDataset(
        split="test",
        config=config
    )

    collate_fn = partial(
        collate_iam_batch,
        image_height=dataset_config["image_height"],
        horizontal_stride=dataset_config["horizontal_stride"],
        background_value=dataset_config.get(
            "background_value", 1.0
        ),
        image_channels=dataset_config["image_channels"]
    )

    test_batch_sampler = WidthBucketBatchSampler(
        dataset=test_dataset,
        batch_size=training_config["batch_size"],
        bucket_size_multiplier=50,
        shuffle=False,
        drop_last=False,
        seed=training_config.get("seed", 42),
        image_height=dataset_config["image_height"]
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=hardware_config["num_workers"],
        pin_memory=hardware_config["pin_memory"],
        collate_fn=collate_fn
    )

    return test_loader


def load_best_model(config, device):
    """Load the best trained SHIE checkpoint."""

    checkpoint_path = resolve_project_path(
        config["checkpoint"]["directory"]
    ) / "best_model.pth"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best SHIE checkpoint not found: {checkpoint_path}"
        )

    model = SHIEModel(
        num_classes=config["model"]["num_classes"],
        config=config
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    print("\nBest SHIE checkpoint loaded successfully.")
    print("Checkpoint:", checkpoint_path)
    print("Best model epoch:", checkpoint.get("epoch"))
    print(
        "Validation CER:",
        checkpoint.get("validation_cer")
    )

    return model, checkpoint


def save_predictions(output_path, sample_ids, references, predictions):
    """Save individual recognition results as CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "reference",
                "prediction",
                "cer",
                "wer"
            ]
        )

        writer.writeheader()

        for sample_id, reference, prediction in zip(
            sample_ids,
            references,
            predictions
        ):

            writer.writerow({
                "sample_id": sample_id,
                "reference": reference,
                "prediction": prediction,
                "cer": SHIEEvaluator.calculate_cer(
                    reference, prediction
                ),
                "wer": SHIEEvaluator.calculate_wer(
                    reference, prediction
                )
            })

    print("\nPrediction report saved:", output_path)


def evaluate_shie(config=None):
    """Evaluate the best SHIE model on the untouched IAM test set."""

    print("\n========== SUTRAYA SHIE TESTING ==========\n")

    # 1. Load configuration.
    if config is None:
        config = load_config()
    else:
        from src.shie.config import validate_config
        validate_config(config)

    # 2. Resolve hardware.
    device = resolve_device(
        config["hardware"]["device"]
    )

    print("Evaluation device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # 3. Load test dataset.
    print("\nLoading IAM test dataset...")

    test_loader = create_test_dataloader(config)

    print("Test samples:", len(test_loader.dataset))
    print("Test batches:", len(test_loader))

    # 4. Load best trained model.
    model, checkpoint = load_best_model(
        config,
        device
    )

    # 5. Initialize decoder and CTC loss.
    decoder = CTCDecoder(
        config["dataset"]["vocabulary_path"]
    )

    criterion = SHIECTCLoss(
        blank_index=config["training"]["blank_index"]
    ).to(device)

    evaluator = SHIEEvaluator()

    # 6. Prepare evaluation storage.
    references = []
    predictions = []
    sample_ids = []

    total_loss = 0.0
    total_samples = 0

    print("\nStarting test evaluation...\n")

    # 7. Run inference without gradient calculation.
    with torch.inference_mode():

        for batch_index, batch in enumerate(test_loader):

            images = batch["images"].to(
                device,
                non_blocking=(device.type == "cuda")
            )

            targets = batch["targets"]
            input_lengths = batch["input_lengths"]
            target_lengths = batch["target_lengths"]

            batch_ids = batch["sample_ids"]
            batch_references = batch["texts"]

            # Forward propagation.
            logits = model(
                images,
                input_lengths=input_lengths
            )

            # Calculate test CTC loss.
            loss = criterion(
                logits,
                targets,
                input_lengths,
                target_lengths
            )

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite test loss in batch {batch_index}."
                )

            # Decode predicted character sequences.
            batch_predictions = decoder.decode_batch(
                logits,
                input_lengths=input_lengths
            )

            # Store results.
            references.extend(batch_references)
            predictions.extend(batch_predictions)
            sample_ids.extend(batch_ids)

            batch_size = len(batch_ids)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if (batch_index + 1) % 25 == 0:
                print(
                    f"Processed {total_samples}/"
                    f"{len(test_loader.dataset)} samples"
                )

    if total_samples == 0:
        raise RuntimeError(
            "Test DataLoader produced no samples."
        )

    # 8. Calculate aggregate CER.
    cer_metrics = evaluator.evaluate_cer_dataset(
        references,
        predictions
    )

    # 9. Calculate aggregate WER.
    wer_metrics = evaluator.evaluate_wer_dataset(
        references,
        predictions
    )

    average_loss = total_loss / total_samples

    # 10. Prepare final metrics.
    results = {
        "project": "SUTRAYA",
        "module": "SHIE",
        "dataset": "IAM",
        "split": "test",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "test_samples": total_samples,
        "test_loss": average_loss,
        "cer": cer_metrics["error_rate"],
        "wer": wer_metrics["error_rate"],
        "cer_statistics": cer_metrics,
        "wer_statistics": wer_metrics
    }

    # 11. Save prediction report.
    output_dir = resolve_project_path(
        "experiments/shie/evaluation"
    )

    predictions_path = (
        output_dir / "test_predictions.csv"
    )

    save_predictions(
        predictions_path,
        sample_ids,
        references,
        predictions
    )

    # 12. Save evaluation metrics.
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_path = output_dir / "test_metrics.json"

    with metrics_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False
        )

    # 13. Display final results.
    print("\n" + "=" * 55)
    print("SHIE TEST EVALUATION RESULTS")
    print("=" * 55)

    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Test samples:", total_samples)
    print(f"Test loss: {average_loss:.6f}")

    print(
        f"CER: {cer_metrics['error_rate']:.4%}"
        if cer_metrics["error_rate"] is not None
        else "CER: N/A"
    )

    print(
        f"WER: {wer_metrics['error_rate']:.4%}"
        if wer_metrics["error_rate"] is not None
        else "WER: N/A"
    )

    print("\nCharacter Error Statistics:")
    print(cer_metrics)

    print("\nWord Error Statistics:")
    print(wer_metrics)

    print("\nMetrics saved:", metrics_path)

    print("\n========== SHIE TESTING COMPLETED ==========")

    return results


if __name__ == "__main__":
    evaluate_shie()
