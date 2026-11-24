"""
SHIE Preprocessing Comparison Tool

Run from the SUTRAYA project root:
    python -m src.shie.preprocessing_comparison

Compares four image-preprocessing variants for the same CTC-aligned character
regions. This is a visual diagnostic only; it does not retrain or evaluate the
recognition model.

Inputs:
    experiments/shie/character_crop_analysis/crop_index.csv

Outputs:
    experiments/shie/preprocessing_comparison/
        pair_*/samples/*_comparison.png
        pair_*/contact_sheet.png
        comparison_index.csv
        summary.json
"""
from pathlib import Path
from collections import defaultdict
import json
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.shie.config import load_config, resolve_project_path
from src.shie.dataset import IAMDataset


def safe_name(value):
    if value == " ":
        return "space"
    if value == "":
        return "none"
    return re.sub(
        r"[^A-Za-z0-9_-]+",
        lambda match: "_".join(f"u{ord(char):04x}" for char in match.group()),
        value,
    )


def resize_grayscale(original, target_height):
    """Resize using the same aspect-ratio calculation as IAMDataset."""
    original_height, original_width = original.shape
    if original_height <= 0 or original_width <= 0:
        raise ValueError("Invalid source image dimensions.")

    target_width = max(
        1,
        round(original_width * target_height / original_height)
    )

    area = cv2.resize(
        original,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    cubic = cv2.resize(
        original,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return area, cubic


def make_variants(original, target_height):
    """Return comparable uint8 grayscale variants at identical dimensions."""
    area, cubic = resize_grayscale(original, target_height)

    # CLAHE enhances local contrast while retaining grayscale information.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    contrast = clahe.apply(area)

    # Adaptive thresholding is included as an experimental visual condition.
    # It is not assumed to be better for cursive handwriting.
    threshold = cv2.adaptiveThreshold(
        area,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )

    return {
        "A_inter_area_baseline": area,
        "B_inter_cubic": cubic,
        "C_clahe_contrast": contrast,
        "D_adaptive_threshold": threshold,
    }


def make_comparison_image(variants, title, output_path, upscale=4):
    """Create a single image with all four variants and clear labels."""
    labels = {
        "A_inter_area_baseline": "A: INTER_AREA (baseline)",
        "B_inter_cubic": "B: INTER_CUBIC",
        "C_clahe_contrast": "C: CLAHE contrast",
        "D_adaptive_threshold": "D: Adaptive threshold",
    }

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    panels = []
    for key, array in variants.items():
        image = Image.fromarray(array, mode="L").convert("RGB")
        image = image.resize(
            (image.width * upscale, image.height * upscale),
            Image.Resampling.LANCZOS,
        )
        panels.append((key, image))

    margin = 12
    label_height = 30
    panel_width = max(image.width for _, image in panels)
    panel_height = max(image.height for _, image in panels)
    canvas_width = margin * 2 + panel_width * 2 + margin
    canvas_height = margin * 2 + (panel_height + label_height) * 2 + margin + 28

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 4), title, fill="black", font=font)

    for index, (key, image) in enumerate(panels):
        col = index % 2
        row = index // 2
        x = margin + col * (panel_width + margin)
        y = margin + 26 + row * (panel_height + label_height + margin)
        canvas.paste(image, (x, y + label_height))
        draw.text((x, y), labels[key], fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def make_contact_sheet(comparison_paths, labels, output_path,
                       columns=2, cell_width=620, cell_height=360):
    if not comparison_paths:
        return

    rows = (len(comparison_paths) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for index, (path, label) in enumerate(zip(comparison_paths, labels)):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail(
                (cell_width - 12, cell_height - 34),
                Image.Resampling.LANCZOS,
            )

        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(image, (x + 6, y + 4))
        draw.text(
            (x + 6, y + cell_height - 25),
            label[:70],
            fill="black",
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main():
    config = load_config()
    dataset = IAMDataset(split="validation", config=config)

    crop_index_path = resolve_project_path(
        "experiments/shie/character_crop_analysis/crop_index.csv"
    )
    output_dir = resolve_project_path(
        "experiments/shie/preprocessing_comparison"
    )

    if not crop_index_path.is_file():
        raise FileNotFoundError(
            f"Missing crop index: {crop_index_path}. "
            "Run character_crop_analyzer first."
        )

    index = pd.read_csv(crop_index_path, keep_default_na=False)
    required = {
        "reference_character", "predicted_character", "sample_id",
        "start_frame", "end_frame", "mean_target_log_probability",
    }
    missing = required - set(index.columns)
    if missing:
        raise ValueError(
            f"Crop index is missing columns: {sorted(missing)}"
        )

    image_height = int(config["dataset"]["image_height"])
    horizontal_stride = int(config["dataset"]["horizontal_stride"])
    context_frames = 4
    max_examples_per_pair = 20

    sample_id_to_row = {
        str(row["sample_id"]): row
        for _, row in dataset.data.iterrows()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    pair_summaries = []

    grouped = index.groupby(
        ["reference_character", "predicted_character"],
        sort=False,
    )

    for (reference_char, predicted_char), group in grouped:
        pair_name = (
            f"pair_{safe_name(str(reference_char))}_to_"
            f"{safe_name(str(predicted_char))}"
        )
        pair_dir = output_dir / pair_name
        samples_dir = pair_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        group = group.copy()
        group["score_numeric"] = pd.to_numeric(
            group["mean_target_log_probability"],
            errors="coerce",
        )
        group = group.sort_values(
            "score_numeric",
            ascending=True,
            na_position="last",
        ).head(max_examples_per_pair)

        comparison_paths = []
        comparison_labels = []
        saved = 0
        skipped = 0

        for row in group.itertuples(index=False):
            sample_id = str(row.sample_id)
            annotation = sample_id_to_row.get(sample_id)

            if annotation is None:
                skipped += 1
                continue

            source_path = dataset._resolve_path(annotation["image_path"])
            original = cv2.imread(
                str(source_path),
                cv2.IMREAD_GRAYSCALE,
            )
            if original is None:
                skipped += 1
                continue

            variants = make_variants(original, image_height)
            baseline_width = variants["A_inter_area_baseline"].shape[1]

            start_frame = int(float(row.start_frame))
            end_frame = int(float(row.end_frame))

            x1 = max(
                0,
                (start_frame - context_frames) * horizontal_stride,
            )
            x2 = min(
                baseline_width,
                (end_frame + 1 + context_frames) * horizontal_stride,
            )
            if x2 <= x1:
                skipped += 1
                continue

            cropped_variants = {
                name: image[:, x1:x2]
                for name, image in variants.items()
            }

            title = (
                f"{sample_id} | {reference_char!r} -> "
                f"{predicted_char!r} | frames {start_frame}-{end_frame}"
            )
            output_path = (
                samples_dir
                / f"{sample_id}_frames_{start_frame}_{end_frame}.png"
            )
            make_comparison_image(
                cropped_variants,
                title,
                output_path,
            )

            comparison_paths.append(output_path)
            comparison_labels.append(
                f"{sample_id}: {reference_char!r}->{predicted_char!r}"
            )

            result_rows.append({
                "reference_character": reference_char,
                "predicted_character": predicted_char,
                "sample_id": sample_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "mean_target_log_probability":
                    row.mean_target_log_probability,
                "source_image": str(source_path),
                "comparison_image": str(output_path),
                "variants": (
                    "INTER_AREA; INTER_CUBIC; CLAHE; adaptive threshold"
                ),
                "note": (
                    "All variants use the same target height and crop x-range. "
                    "The crop position is approximate CTC alignment."
                ),
            })
            saved += 1

        contact_sheet = pair_dir / "contact_sheet.png"
        make_contact_sheet(
            comparison_paths,
            comparison_labels,
            contact_sheet,
        )

        pair_summaries.append({
            "reference_character": reference_char,
            "predicted_character": predicted_char,
            "examples_requested": len(group),
            "comparisons_saved": saved,
            "skipped": skipped,
            "contact_sheet": str(contact_sheet),
        })

        print(
            f"{reference_char!r} -> {predicted_char!r}: "
            f"{saved} comparisons saved; {skipped} skipped"
        )

    output_index = output_dir / "comparison_index.csv"
    pd.DataFrame(result_rows).to_csv(
        output_index,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "validation_split_only": True,
        "model_retraining_performed": False,
        "image_height": image_height,
        "horizontal_stride": horizontal_stride,
        "context_frames_each_side": context_frames,
        "variants": [
            {
                "name": "A_inter_area_baseline",
                "description": "Current IAMDataset resize interpolation.",
            },
            {
                "name": "B_inter_cubic",
                "description": "Cubic interpolation at the same target dimensions.",
            },
            {
                "name": "C_clahe_contrast",
                "description": "CLAHE applied to the INTER_AREA resized grayscale image.",
            },
            {
                "name": "D_adaptive_threshold",
                "description": "Adaptive Gaussian threshold applied to the INTER_AREA image.",
            },
        ],
        "pairs": pair_summaries,
        "important_note": (
            "This is a visual comparison, not an accuracy experiment. "
            "No model predictions or CER/WER are computed for alternate variants. "
            "CTC crop coordinates are approximate, not ground-truth boxes."
        ),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSHIE preprocessing comparison completed.")
    print(f"Comparison index: {output_index}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
