
from pathlib import Path
import textwrap

import cv2
import pandas as pd
import matplotlib.pyplot as plt

from src.shie.config import (
    load_config,
    resolve_project_path,
)


TOP_K = 20


def analyze_errors():

    print("\n========== SHIE ERROR ANALYSIS ==========\n")

    # -------------------------------------------------
    # 1. Load configuration and resolve file paths
    # -------------------------------------------------

    config = load_config()

    image_height = config["dataset"]["image_height"]

    test_csv_path = resolve_project_path(
        config["dataset"]["test_csv"]
    )

    predictions_path = resolve_project_path(
        "experiments/shie/evaluation/test_predictions.csv"
    )

    output_dir = resolve_project_path(
        "experiments/shie/error_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not test_csv_path.is_file():
        raise FileNotFoundError(test_csv_path)

    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)

    # -------------------------------------------------
    # 2. Load test annotations and model predictions
    # -------------------------------------------------

    print("Loading test annotations...")
    annotations = pd.read_csv(
        test_csv_path,
        dtype={"sample_id": str},
        keep_default_na=False
    )

    print("Loading SHIE predictions...")
    predictions = pd.read_csv(
        predictions_path,
        dtype={"sample_id": str},
        keep_default_na=False
    )

    required_annotation_columns = {
        "sample_id",
        "image_path",
        "text"
    }

    required_prediction_columns = {
        "sample_id",
        "reference",
        "prediction",
        "cer",
        "wer"
    }

    if not required_annotation_columns.issubset(
        annotations.columns
    ):
        raise ValueError(
            "Test annotation CSV is missing required columns."
        )

    if not required_prediction_columns.issubset(
        predictions.columns
    ):
        raise ValueError(
            "Prediction CSV is missing required columns."
        )

    # -------------------------------------------------
    # 3. Check sample IDs
    # -------------------------------------------------

    if annotations["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample IDs found in test annotations."
        )

    if predictions["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample IDs found in predictions."
        )

    merged = annotations.merge(
        predictions,
        on="sample_id",
        how="inner",
        validate="one_to_one"
    )

    if len(merged) != len(annotations):
        raise ValueError(
            "Not all test annotations have matching predictions. "
            f"Annotations: {len(annotations)}, "
            f"Matched: {len(merged)}"
        )

    if len(merged) != len(predictions):
        raise ValueError(
            "Some predictions do not match test annotations."
        )

    print(f"Successfully matched: {len(merged)} samples")

    # -------------------------------------------------
    # 4. Verify reference transcription integrity
    # -------------------------------------------------

    mismatches = merged[
        merged["text"] != merged["reference"]
    ]

    if not mismatches.empty:

        mismatch_path = output_dir / "reference_mismatches.csv"

        mismatches[
            [
                "sample_id",
                "text",
                "reference",
                "prediction"
            ]
        ].to_csv(
            mismatch_path,
            index=False
        )

        raise ValueError(
            f"{len(mismatches)} reference mismatches found. "
            f"See: {mismatch_path}"
        )

    print("Reference transcription verification passed.")

    # -------------------------------------------------
    # 5. Rank difficult samples
    # -------------------------------------------------

    merged["cer"] = pd.to_numeric(
        merged["cer"],
        errors="raise"
    )

    merged["wer"] = pd.to_numeric(
        merged["wer"],
        errors="raise"
    )

    worst_samples = merged.sort_values(
        by=["cer", "wer"],
        ascending=[False, False]
    ).head(TOP_K)

    # Save selected error cases.
    report_path = output_dir / "worst_samples.csv"

    worst_samples[
        [
            "sample_id",
            "image_path",
            "text",
            "reference",
            "prediction",
            "cer",
            "wer"
        ]
    ].to_csv(
        report_path,
        index=False
    )

    print(f"\nSelected {len(worst_samples)} difficult samples.")

    # -------------------------------------------------
    # 6. Generate visual comparisons
    # -------------------------------------------------

    for rank, (_, row) in enumerate(
        worst_samples.iterrows(),
        start=1
    ):

        sample_id = row["sample_id"]

        image_path = resolve_project_path(
            row["image_path"]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found for {sample_id}: "
                f"{image_path}"
            )

        # Load original IAM handwriting image.
        original_image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if original_image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        original_height, original_width = (
            original_image.shape
        )

        # Reproduce the model's height-based resizing.
        resized_width = max(
            1,
            round(
                original_width
                * image_height
                / original_height
            )
        )

        resized_image = cv2.resize(
            original_image,
            (resized_width, image_height),
            interpolation=cv2.INTER_AREA
        )

        # -------------------------------------------------
        # Display original and resized model input
        # -------------------------------------------------

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(16, 4)
        )

        axes[0].imshow(
            original_image,
            cmap="gray"
        )

        axes[0].set_title(
            "Original IAM Image"
        )

        axes[1].imshow(
            resized_image,
            cmap="gray"
        )

        axes[1].set_title(
            "Resized Model Input"
        )

        for axis in axes:
            axis.axis("off")

        fig.suptitle(
            f"{sample_id} | "
            f"CER: {row['cer']:.4f} | "
            f"WER: {row['wer']:.4f}",
            fontsize=14,
            fontweight="bold"
        )

        reference_text = textwrap.fill(
            str(row["text"]),
            width=115
        )

        prediction_text = textwrap.fill(
            str(row["prediction"]),
            width=115
        )

        fig.text(
            0.05,
            0.17,
            f"REFERENCE: {reference_text}",
            fontsize=10,
            verticalalignment="top"
        )

        fig.text(
            0.05,
            0.07,
            f"PREDICTION: {prediction_text}",
            fontsize=10,
            verticalalignment="top"
        )

        plt.tight_layout(
            rect=[0, 0.25, 1, 0.90]
        )

        safe_sample_id = sample_id.replace(
            "/",
            "_"
        ).replace(
            "\\",
            "_"
        )

        output_path = output_dir / (
            f"{rank:02d}_{safe_sample_id}_"
            f"cer_{row['cer']:.4f}.png"
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"[{rank}/{len(worst_samples)}] "
            f"Saved: {output_path.name}"
        )

    # -------------------------------------------------
    # 7. Finish
    # -------------------------------------------------

    print("\n========== ERROR ANALYSIS COMPLETED ==========")

    print(f"Matched samples: {len(merged)}")
    print(f"Inspected samples: {len(worst_samples)}")

    print(f"\nOutput directory: {output_dir}")
    print(f"Error report: {report_path}")

    print("\nOpen the generated PNG files to inspect")
    print("the original handwriting and SHIE predictions.")


if __name__ == "__main__":
    analyze_errors()
