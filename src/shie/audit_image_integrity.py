
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd

from src.shie.config import load_config, resolve_project_path


# --------------------------------------------------
# Audit thresholds
# --------------------------------------------------

# Images with less than this proportion of dark pixels
# are candidates for blank-image inspection.
MIN_FOREGROUND_RATIO = 0.001

# Extremely wide or narrow images are flagged for review.
MIN_ASPECT_RATIO = 2.0
MAX_ASPECT_RATIO = 30.0

# Nearly uniform images are suspicious.
MIN_PIXEL_STD = 3.0


def analyze_image(image_path, expected_width, expected_height):

    result = {
        "image_exists": False,
        "image_readable": False,
        "actual_width": None,
        "actual_height": None,
        "expected_width": expected_width,
        "expected_height": expected_height,
        "aspect_ratio": None,
        "pixel_mean": None,
        "pixel_std": None,
        "foreground_ratio": None,
        "file_size_bytes": None,
        "issues": []
    }

    # --------------------------------------------------
    # 1. Check file existence
    # --------------------------------------------------

    if not image_path.is_file():
        result["issues"].append("missing_image")
        return result

    result["image_exists"] = True

    result["file_size_bytes"] = image_path.stat().st_size

    if result["file_size_bytes"] == 0:
        result["issues"].append("empty_image_file")
        return result

    # --------------------------------------------------
    # 2. Decode image
    # --------------------------------------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        result["issues"].append("unreadable_or_corrupt_image")
        return result

    result["image_readable"] = True

    height, width = image.shape

    result["actual_width"] = int(width)
    result["actual_height"] = int(height)

    # --------------------------------------------------
    # 3. Validate dimensions
    # --------------------------------------------------

    if width <= 0 or height <= 0:
        result["issues"].append("invalid_dimensions")
        return result

    if width != expected_width or height != expected_height:
        result["issues"].append("metadata_dimension_mismatch")

    # --------------------------------------------------
    # 4. Calculate image statistics
    # --------------------------------------------------

    result["aspect_ratio"] = round(
        width / height,
        4
    )

    result["pixel_mean"] = round(
        float(np.mean(image)),
        4
    )

    result["pixel_std"] = round(
        float(np.std(image)),
        4
    )

    # In our preprocessing convention:
    # White background = 255
    # Dark handwriting = lower pixel values.
    foreground_pixels = np.count_nonzero(
        image < 245
    )

    foreground_ratio = (
        foreground_pixels / image.size
    )

    result["foreground_ratio"] = round(
        float(foreground_ratio),
        6
    )

    # --------------------------------------------------
    # 5. Identify suspicious image characteristics
    # --------------------------------------------------

    if result["pixel_std"] < MIN_PIXEL_STD:
        result["issues"].append("nearly_uniform_image")

    if foreground_ratio < MIN_FOREGROUND_RATIO:
        result["issues"].append("possible_blank_image")

    if result["aspect_ratio"] < MIN_ASPECT_RATIO:
        result["issues"].append("unusually_narrow_image")

    if result["aspect_ratio"] > MAX_ASPECT_RATIO:
        result["issues"].append("unusually_wide_image")

    return result


def main():

    print("\n========== SHIE IMAGE INTEGRITY AUDIT ==========\n")

    config = load_config()

    dataset_config = config["dataset"]

    splits = {
        "train": dataset_config["train_csv"],
        "validation": dataset_config["validation_csv"],
        "test": dataset_config["test_csv"]
    }

    output_dir = resolve_project_path(
        "experiments/shie/dataset_audit"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    # --------------------------------------------------
    # 1. Inspect all three splits
    # --------------------------------------------------

    for split, configured_csv in splits.items():

        csv_path = resolve_project_path(configured_csv)

        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)

        print(f"\nInspecting {split} split...")
        print("CSV:", csv_path)

        df = pd.read_csv(
            csv_path,
            keep_default_na=False
        )

        required_columns = {
            "sample_id",
            "image_path",
            "width",
            "height"
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{split} CSV missing columns: {sorted(missing)}"
            )

        split_results = []

        for _, row in df.iterrows():

            image_path = resolve_project_path(
                row["image_path"]
            )

            expected_width = int(row["width"])
            expected_height = int(row["height"])

            metrics = analyze_image(
                image_path=image_path,
                expected_width=expected_width,
                expected_height=expected_height
            )

            record = {
                "split": split,
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                **metrics
            }

            split_results.append(record)
            all_results.append(record)

        # --------------------------------------------------
        # 2. Split summary
        # --------------------------------------------------

        split_df = pd.DataFrame(split_results)

        flagged = split_df[
            split_df["issues"].map(bool)
        ]

        missing_count = (
            split_df["image_exists"] == False
        ).sum()

        unreadable_count = (
            (split_df["image_exists"] == True)
            & (split_df["image_readable"] == False)
        ).sum()

        print("Total images:", len(split_df))
        print("Missing images:", missing_count)
        print("Unreadable images:", unreadable_count)
        print("Flagged images:", len(flagged))

    # --------------------------------------------------
    # 3. Save complete metrics and flagged records
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)

    flagged_df = results_df[
        results_df["issues"].map(bool)
    ]

    complete_report = output_dir / "image_integrity_all.csv"

    flagged_report = output_dir / "image_integrity_issues.csv"

    results_df.to_csv(
        complete_report,
        index=False
    )

    flagged_df.to_csv(
        flagged_report,
        index=False
    )

    # --------------------------------------------------
    # 4. Generate JSON summary
    # --------------------------------------------------

    summary = {
        "project": "SUTRAYA",
        "module": "SHIE",
        "total_images": len(results_df),
        "total_flagged": len(flagged_df),
        "splits": {}
    }

    for split in splits:

        split_df = results_df[
            results_df["split"] == split
        ]

        split_flagged = split_df[
            split_df["issues"].map(bool)
        ]

        summary["splits"][split] = {
            "total": len(split_df),
            "missing": int(
                (~split_df["image_exists"]).sum()
            ),
            "unreadable": int(
                (
                    split_df["image_exists"]
                    & ~split_df["image_readable"]
                ).sum()
            ),
            "flagged": len(split_flagged)
        }

    summary_path = output_dir / "image_integrity_summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # --------------------------------------------------
    # 5. Final report
    # --------------------------------------------------

    print("\n========== FINAL AUDIT SUMMARY ==========")

    print("Total images inspected:", len(results_df))
    print("Total flagged images:", len(flagged_df))

    for split, stats in summary["splits"].items():

        print(f"\n{split.upper()}")

        print("Total:", stats["total"])
        print("Missing:", stats["missing"])
        print("Unreadable:", stats["unreadable"])
        print("Flagged:", stats["flagged"])

    print("\nReports saved:")
    print(complete_report)
    print(flagged_report)
    print(summary_path)

    print("\nIMPORTANT:")
    print("Flagged images are candidates for manual inspection.")
    print("No images or annotations were modified.")

    print("\n========== IMAGE AUDIT COMPLETED ==========")


if __name__ == "__main__":
    main()
