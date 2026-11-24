"""
SHIE Character Crop Analyzer — clean-image version.

Run from the SUTRAYA project root:
    python -m src.shie.character_crop_analyzer

This version obtains images directly from IAMDataset, so crops contain the
dataset-preprocessed grayscale handwriting without red annotation overlays.

CTC frame-to-image coordinates are approximate, not ground-truth character boxes.
"""
from pathlib import Path
from collections import Counter
import json
import re

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from src.shie.config import load_config, resolve_project_path
from src.shie.dataset import IAMDataset


def safe_name(value):
    """Create a filesystem-safe name for a character."""
    if value == " ":
        return "space"
    if value == "":
        return "none"
    return re.sub(
        r"[^A-Za-z0-9_-]+",
        lambda match: "_".join(f"u{ord(char):04x}" for char in match.group()),
        value,
    )


def make_contact_sheet(crop_paths, labels, output_path,
                       columns=4, cell_width=260, cell_height=170):
    """Create a labeled overview image from individual clean crops."""
    if not crop_paths:
        return

    rows = (len(crop_paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for index, (crop_path, label) in enumerate(zip(crop_paths, labels)):
        with Image.open(crop_path) as source:
            image = source.convert("RGB")
            image.thumbnail(
                (cell_width - 16, cell_height - 48),
                Image.Resampling.LANCZOS,
            )

        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        image_x = x + (cell_width - image.width) // 2
        sheet.paste(image, (image_x, y + 5))
        draw.text(
            (x + 8, y + cell_height - 38),
            label[:48],
            fill="black",
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def tensor_to_clean_image(image_tensor):
    """
    Convert IAMDataset's [1,H,W] float tensor into a clean grayscale PIL image.

    Dataset normalization is white=1.0 and dark ink=0.0.
    """
    pixels = (
        image_tensor.detach()
        .cpu()
        .squeeze(0)
        .clamp(0.0, 1.0)
        .mul(255)
        .round()
        .byte()
        .numpy()
    )
    return Image.fromarray(pixels, mode="L")


def main():
    config = load_config()
    dataset = IAMDataset(split="validation", config=config)

    analysis_dir = resolve_project_path("experiments/shie/visual_error_analysis")
    alignment_csv = analysis_dir / "visual_error_alignment.csv"
    output_dir = resolve_project_path("experiments/shie/character_crop_analysis")

    if not alignment_csv.is_file():
        raise FileNotFoundError(f"Missing alignment CSV: {alignment_csv}")

    horizontal_stride = int(config["dataset"]["horizontal_stride"])
    max_examples_per_pair = 20
    context_frames = 4
    upscale_factor = 4

    data = pd.read_csv(alignment_csv, keep_default_na=False)
    required_columns = {
        "sample_id", "operation", "reference_char", "prediction_char",
        "reference_index", "prediction_index", "start_frame", "end_frame",
        "mean_target_log_probability", "reference", "prediction",
    }
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Alignment CSV is missing columns: {sorted(missing)}")

    substitutions = data[data["operation"] == "substitution"].copy()
    pair_counts = Counter(
        (str(row.reference_char), str(row.prediction_char))
        for row in substitutions.itertuples(index=False)
    )
    top_pairs = pair_counts.most_common(5)

    # Build a direct lookup into the same validation dataset used for alignment.
    # This avoids reading the previously annotated PNG files.
    sample_id_to_index = {
        str(sample_id): index
        for index, sample_id in enumerate(dataset.data["sample_id"].tolist())
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_index_rows = []
    summary_pairs = []

    for (reference_char, predicted_char), pair_count in top_pairs:
        pair_name = (
            f"pair_{safe_name(reference_char)}_to_{safe_name(predicted_char)}"
        )
        pair_dir = output_dir / pair_name
        crops_dir = pair_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        pair_rows = substitutions[
            (substitutions["reference_char"] == reference_char)
            & (substitutions["prediction_char"] == predicted_char)
        ].copy()

        pair_rows["start_numeric"] = pd.to_numeric(
            pair_rows["start_frame"], errors="coerce"
        )
        pair_rows["end_numeric"] = pd.to_numeric(
            pair_rows["end_frame"], errors="coerce"
        )
        pair_rows["score_numeric"] = pd.to_numeric(
            pair_rows["mean_target_log_probability"], errors="coerce"
        )
        pair_rows = pair_rows.dropna(
            subset=["start_numeric", "end_numeric"]
        ).sort_values(
            "score_numeric", ascending=True, na_position="last"
        ).head(max_examples_per_pair)

        crop_paths = []
        crop_labels = []
        saved_count = 0
        skipped_missing_sample = 0

        for row in pair_rows.itertuples(index=False):
            sample_id = str(row.sample_id)
            dataset_index = sample_id_to_index.get(sample_id)

            if dataset_index is None:
                skipped_missing_sample += 1
                continue

            # Reuse the exact preprocessing implementation in IAMDataset.
            sample = dataset[dataset_index]
            clean_line = tensor_to_clean_image(sample["image"])
            width, height = clean_line.size

            start_frame = int(float(row.start_frame))
            end_frame = int(float(row.end_frame))

            # Each model timestep corresponds approximately to horizontal_stride
            # input pixels. Include neighboring frames for connected handwriting.
            x1 = max(
                0,
                (start_frame - context_frames) * horizontal_stride,
            )
            x2 = min(
                width,
                (end_frame + 1 + context_frames) * horizontal_stride,
            )

            if x2 <= x1:
                continue

            crop = clean_line.crop((x1, 0, x2, height))
            crop = crop.resize(
                (max(1, crop.width * upscale_factor),
                 max(1, crop.height * upscale_factor)),
                Image.Resampling.LANCZOS,
            )

            reference_index = int(float(row.reference_index))
            prediction_index = int(float(row.prediction_index))
            crop_filename = (
                f"{sample_id}_ref{reference_index}_pred{prediction_index}.png"
            )
            crop_path = crops_dir / crop_filename
            crop.save(crop_path)

            score = row.mean_target_log_probability
            label = (
                f"{sample_id} | {reference_char!r}->{predicted_char!r} "
                f"| logp={score}"
            )
            crop_paths.append(crop_path)
            crop_labels.append(label)

            crop_index_rows.append({
                "reference_character": reference_char,
                "predicted_character": predicted_char,
                "pair_count_in_validation": pair_count,
                "sample_id": sample_id,
                "reference_index": reference_index,
                "prediction_index": prediction_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "mean_target_log_probability": score,
                "crop_path": str(crop_path),
                "reference": row.reference,
                "prediction": row.prediction,
                "source": "IAMDataset validation preprocessing",
                "note": (
                    "Clean grayscale crop with neighboring handwriting. "
                    "Horizontal position is approximate CTC alignment, "
                    "not a ground-truth character box."
                ),
            })
            saved_count += 1

        contact_sheet = pair_dir / "contact_sheet.png"
        make_contact_sheet(crop_paths, crop_labels, contact_sheet)

        summary_pairs.append({
            "reference_character": reference_char,
            "predicted_character": predicted_char,
            "total_pair_occurrences": pair_count,
            "crops_saved": saved_count,
            "skipped_missing_sample": skipped_missing_sample,
            "contact_sheet": str(contact_sheet),
        })

        print(
            f"{reference_char!r} -> {predicted_char!r}: "
            f"{saved_count} clean crops saved (of {pair_count} occurrences)"
        )

    index_path = output_dir / "crop_index.csv"
    pd.DataFrame(crop_index_rows).to_csv(
        index_path, index=False, encoding="utf-8-sig"
    )

    summary = {
        "validation_samples_in_dataset": len(dataset),
        "validation_substitution_rows": len(substitutions),
        "pairs_analyzed": len(top_pairs),
        "max_examples_per_pair": max_examples_per_pair,
        "context_frames_each_side": context_frames,
        "horizontal_stride": horizontal_stride,
        "upscale_factor": upscale_factor,
        "source_images": "IAMDataset.__getitem__ validation preprocessing",
        "pairs": summary_pairs,
        "important_note": (
            "Crops are generated from clean dataset tensors, not annotated PNGs. "
            "CTC frame-to-image positions remain approximate because model "
            "receptive fields are wider than a single stride interval. "
            "These are not ground-truth character bounding boxes."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSHIE clean character crop analysis completed.")
    print(f"Crop index: {index_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
