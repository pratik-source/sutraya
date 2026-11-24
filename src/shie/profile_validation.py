
import json
from functools import partial
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.shie.config import load_config, resolve_project_path
from src.shie.dataset import IAMDataset, collate_iam_batch
from src.shie.bucket_sampler import WidthBucketBatchSampler
from src.shie.model import SHIEModel
from src.shie.decoder import CTCDecoder
from src.shie.evaluator import SHIEEvaluator


# ==========================================================
# DEVICE
# ==========================================================

def resolve_device(device_config):

    if device_config == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_config == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        return torch.device("cuda")

    return torch.device("cpu")


# ==========================================================
# VALIDATION DATA LOADER
# ==========================================================

def create_validation_loader(config):

    dataset = IAMDataset(
        split="validation",
        config=config
    )

    dataset_config = config["dataset"]
    training_config = config["training"]
    hardware_config = config["hardware"]

    collate_fn = partial(
        collate_iam_batch,
        image_height=dataset_config["image_height"],
        horizontal_stride=dataset_config["horizontal_stride"],
        background_value=dataset_config.get(
            "background_value", 1.0
        ),
        image_channels=dataset_config["image_channels"],
    )

    sampler = WidthBucketBatchSampler(
        dataset=dataset,
        batch_size=training_config["batch_size"],
        bucket_size_multiplier=50,
        shuffle=False,
        drop_last=False,
        seed=training_config.get("seed", 42),
        image_height=dataset_config["image_height"],
    )

    loader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=hardware_config["num_workers"],
        pin_memory=hardware_config["pin_memory"],
        collate_fn=collate_fn,
    )

    return dataset, loader


# ==========================================================
# LOAD BEST CHECKPOINT
# ==========================================================

def load_best_model(config, device):

    checkpoint_dir = resolve_project_path(
        config["checkpoint"]["directory"]
    )

    checkpoint_path = checkpoint_dir / "best_model.pth"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model = SHIEModel(
        num_classes=config["model"]["num_classes"],
        config=config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    print("Best checkpoint loaded successfully.")
    print("Checkpoint epoch:", checkpoint["epoch"])

    return model, checkpoint


# ==========================================================
# ANALYSIS CATEGORIES
# ==========================================================

def get_width_category(input_length):

    if input_length < 256:
        return "short"

    if input_length < 512:
        return "medium"

    return "long"


def get_text_length_category(length):

    if length <= 25:
        return "short"

    if length <= 50:
        return "medium"

    if length <= 75:
        return "long"

    return "very_long"


# ==========================================================
# GROUP METRICS
# ==========================================================

def calculate_group_metrics(records, group_field):

    grouped = {}

    for record in records:
        key = record[group_field]
        grouped.setdefault(key, []).append(record)

    results = []

    for category, samples in grouped.items():

        references = [
            sample["reference"] for sample in samples
        ]

        predictions = [
            sample["prediction"] for sample in samples
        ]

        cer_stats = SHIEEvaluator.evaluate_cer_dataset(
            references,
            predictions
        )

        wer_stats = SHIEEvaluator.evaluate_wer_dataset(
            references,
            predictions
        )

        results.append({
            "category": category,
            "samples": len(samples),
            "cer": cer_stats["error_rate"],
            "wer": wer_stats["error_rate"],
            "character_errors": cer_stats["total_errors"],
            "word_errors": wer_stats["total_errors"],
            "substitutions": cer_stats["substitutions"],
            "deletions": cer_stats["deletions"],
            "insertions": cer_stats["insertions"],
            "reference_characters": cer_stats["reference_units"],
            "reference_words": wer_stats["reference_units"],
        })

    return results


# ==========================================================
# SAVE WORST SAMPLE IMAGE COMPARISONS
# ==========================================================

def save_sample_comparison(dataset, sample_id, output_dir):

    matching = dataset.data.index[
        dataset.data["sample_id"].astype(str) == str(sample_id)
    ]

    if len(matching) != 1:
        raise ValueError(
            f"Could not uniquely locate sample: {sample_id}"
        )

    index = int(matching[0])
    row = dataset.data.iloc[index]

    original_path = dataset._resolve_path(
        row["image_path"]
    )

    original = cv2.imread(
        str(original_path),
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        raise FileNotFoundError(original_path)

    # Reproduce the exact dataset resizing operation.
    original_height, original_width = original.shape

    resized_width = max(
        1,
        round(
            original_width
            * dataset.image_height
            / original_height
        )
    )

    resized = cv2.resize(
        original,
        (resized_width, dataset.image_height),
        interpolation=cv2.INTER_AREA
    )

    # Enlarge model input for visual inspection.
    resized_display = cv2.resize(
        resized,
        (
            resized_width * 4,
            dataset.image_height * 4
        ),
        interpolation=cv2.INTER_NEAREST
    )

    original_display = cv2.resize(
        original,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    # Match display heights.
    display_height = max(
        original_display.shape[0],
        resized_display.shape[0]
    )

    original_display = cv2.resize(
        original_display,
        (
            round(
                original_display.shape[1]
                * display_height
                / original_display.shape[0]
            ),
            display_height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    resized_display = cv2.resize(
        resized_display,
        (
            round(
                resized_display.shape[1]
                * display_height
                / resized_display.shape[0]
            ),
            display_height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    # Place both images side by side.
    comparison = cv2.hconcat([
        original_display,
        resized_display
    ])

    output_path = output_dir / f"{sample_id}_comparison.png"

    if not cv2.imwrite(str(output_path), comparison):
        raise RuntimeError(
            f"Could not save comparison image: {output_path}"
        )


# ==========================================================
# MAIN VALIDATION PROFILING
# ==========================================================

def profile_validation(config=None):

    print("\n========== SHIE VALIDATION PROFILING ==========\n")

    if config is None:
        config = load_config()

    device = resolve_device(
        config["hardware"]["device"]
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------------------------------------------
    # 1. LOAD DATASET
    # ------------------------------------------------------

    print("\nLoading validation dataset...")

    dataset, loader = create_validation_loader(config)

    print("Validation samples:", len(dataset))
    print("Validation batches:", len(loader))

    # ------------------------------------------------------
    # 2. LOAD MODEL
    # ------------------------------------------------------

    model, checkpoint = load_best_model(
        config,
        device
    )

    decoder = CTCDecoder(
        config["dataset"]["vocabulary_path"]
    )

    # ------------------------------------------------------
    # 3. PREPARE OUTPUT DIRECTORIES
    # ------------------------------------------------------

    output_dir = resolve_project_path(
        "experiments/shie/validation_profiling"
    )

    image_dir = output_dir / "worst_20_images"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    # ------------------------------------------------------
    # 4. RUN INFERENCE
    # ------------------------------------------------------

    print("\nGenerating validation predictions...")

    with torch.inference_mode():

        for batch_index, batch in enumerate(loader, start=1):

            images = batch["images"].to(
                device,
                non_blocking=True
            )

            input_lengths = batch["input_lengths"]

            logits = model(
                images,
                input_lengths=input_lengths
            )

            predictions = decoder.decode_batch(
                logits,
                input_lengths=input_lengths
            )

            references = batch["texts"]
            sample_ids = batch["sample_ids"]

            for index, (reference, prediction) in enumerate(
                zip(references, predictions)
            ):

                input_length = int(
                    input_lengths[index].item()
                )

                target_length = len(reference)

                cer = SHIEEvaluator.calculate_cer(
                    reference,
                    prediction
                )

                wer = SHIEEvaluator.calculate_wer(
                    reference,
                    prediction
                )

                errors = SHIEEvaluator.edit_statistics(
                    list(reference),
                    list(prediction)
                )

                # Locate original annotation dimensions.
                sample_row = dataset.data[
                    dataset.data["sample_id"].astype(str)
                    == str(sample_ids[index])
                ]

                if len(sample_row) != 1:
                    raise ValueError(
                        f"Invalid sample ID: {sample_ids[index]}"
                    )

                original_width = int(
                    sample_row.iloc[0]["width"]
                )

                original_height = int(
                    sample_row.iloc[0]["height"]
                )

                resized_width = max(
                    1,
                    round(
                        original_width
                        * dataset.image_height
                        / original_height
                    )
                )

                records.append({
                    "sample_id": str(sample_ids[index]),
                    "reference": reference,
                    "prediction": prediction,
                    "cer": cer,
                    "wer": wer,
                    "input_length": input_length,
                    "target_length": target_length,
                    "word_count": len(reference.split()),
                    "original_width": original_width,
                    "original_height": original_height,
                    "resized_width": resized_width,
                    "width_category": get_width_category(
                        input_length
                    ),
                    "text_length_category": (
                        get_text_length_category(target_length)
                    ),
                    "substitutions": errors["substitutions"],
                    "deletions": errors["deletions"],
                    "insertions": errors["insertions"],
                    "total_errors": errors["total_errors"],
                })

            if batch_index % 20 == 0 or batch_index == len(loader):
                print(
                    f"Processed {batch_index}/{len(loader)} batches"
                )

    # ------------------------------------------------------
    # 5. VERIFY DATASET COVERAGE
    # ------------------------------------------------------

    if len(records) != len(dataset):
        raise RuntimeError(
            f"Expected {len(dataset)} predictions, "
            f"received {len(records)}."
        )

    prediction_df = pd.DataFrame(records)

    if prediction_df["sample_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate sample IDs found in predictions."
        )

    # ------------------------------------------------------
    # 6. OVERALL METRICS
    # ------------------------------------------------------

    references = prediction_df["reference"].tolist()
    predictions = prediction_df["prediction"].tolist()

    cer_stats = SHIEEvaluator.evaluate_cer_dataset(
        references,
        predictions
    )

    wer_stats = SHIEEvaluator.evaluate_wer_dataset(
        references,
        predictions
    )

    overall = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": "validation",
        "samples": len(records),
        "cer": cer_stats["error_rate"],
        "wer": wer_stats["error_rate"],
        "character_statistics": cer_stats,
        "word_statistics": wer_stats,
    }

    # ------------------------------------------------------
    # 7. WIDTH AND TEXT LENGTH PROFILING
    # ------------------------------------------------------

    width_metrics = calculate_group_metrics(
        records,
        "width_category"
    )

    text_length_metrics = calculate_group_metrics(
        records,
        "text_length_category"
    )

    # ------------------------------------------------------
    # 8. SAVE ERROR BREAKDOWN
    # ------------------------------------------------------

    error_breakdown = {
        "substitutions": int(
            prediction_df["substitutions"].sum()
        ),
        "deletions": int(
            prediction_df["deletions"].sum()
        ),
        "insertions": int(
            prediction_df["insertions"].sum()
        ),
        "total_errors": int(
            prediction_df["total_errors"].sum()
        ),
    }

    # ------------------------------------------------------
    # 9. WORST 20 PREDICTIONS
    # ------------------------------------------------------

    worst_df = prediction_df.sort_values(
        by=["cer", "wer", "total_errors"],
        ascending=[False, False, False]
    ).head(20)

    # ------------------------------------------------------
    # 10. SAVE IMAGE COMPARISONS
    # ------------------------------------------------------

    print("\nSaving worst-sample image comparisons...")

    for sample_id in worst_df["sample_id"]:
        save_sample_comparison(
            dataset,
            sample_id,
            image_dir
        )

    # ------------------------------------------------------
    # 11. SAVE OUTPUT FILES
    # ------------------------------------------------------

    prediction_df.to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(width_metrics).to_csv(
        output_dir / "width_category_metrics.csv",
        index=False
    )

    pd.DataFrame(text_length_metrics).to_csv(
        output_dir / "text_length_metrics.csv",
        index=False
    )

    worst_df.to_csv(
        output_dir / "worst_20_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    results = {
        "project": "SUTRAYA",
        "module": "SHIE",
        "evaluation": "validation_performance_profiling",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "overall": overall,
        "width_category_metrics": width_metrics,
        "text_length_metrics": text_length_metrics,
        "error_breakdown": error_breakdown,
    }

    with (output_dir / "validation_profile.json").open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ------------------------------------------------------
    # 12. PRINT SUMMARY
    # ------------------------------------------------------

    print("\n========== PROFILING COMPLETED ==========")

    print("Samples:", len(records))
    print(f"CER: {overall['cer']:.4%}")
    print(f"WER: {overall['wer']:.4%}")

    print("\nWidth Category Metrics")
    print(pd.DataFrame(width_metrics).to_string(index=False))

    print("\nText Length Metrics")
    print(pd.DataFrame(text_length_metrics).to_string(index=False))

    print("\nCharacter Error Breakdown")
    print(error_breakdown)

    print("\nOutput directory:")
    print(output_dir)

    return results


if __name__ == "__main__":
    profile_validation()
