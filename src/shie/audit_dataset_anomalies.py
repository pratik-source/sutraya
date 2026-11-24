
from pathlib import Path
import re

import pandas as pd

from src.shie.config import load_config, resolve_project_path


def inspect_transcription(text):
    """
    Identify suspicious transcription patterns.
    This function flags candidates for manual inspection.
    It does not decide whether a transcription is incorrect.
    """

    text = str(text).strip()

    reasons = []

    if not text:
        reasons.append("empty_transcription")
        return reasons

    alphanumeric_count = sum(
        char.isalnum() for char in text
    )

    punctuation_ratio = (
        1 - alphanumeric_count / len(text)
    )

    if len(text) >= 10 and alphanumeric_count == 0:
        reasons.append("no_alphanumeric_characters")

    if len(text) >= 10 and punctuation_ratio >= 0.80:
        reasons.append("very_high_punctuation_ratio")

    if re.search(r"(.)\1{9,}", text):
        reasons.append("long_repeated_character_sequence")

    return reasons


def main():

    print("\n========== SHIE DATASET ANOMALY AUDIT ==========\n")

    config = load_config()

    splits = {
        "train": config["dataset"]["train_csv"],
        "validation": config["dataset"]["validation_csv"],
        "test": config["dataset"]["test_csv"],
    }

    output_dir = resolve_project_path(
        "experiments/shie/dataset_audit"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_flags = []

    for split, csv_path in splits.items():

        csv_path = resolve_project_path(csv_path)

        print(f"\nInspecting {split} split...")
        print(f"CSV: {csv_path}")

        df = pd.read_csv(
            csv_path,
            keep_default_na=False
        )

        required = {
            "sample_id",
            "image_path",
            "text"
        }

        if not required.issubset(df.columns):
            raise ValueError(
                f"{split} CSV is missing required columns."
            )

        split_flags = []

        for _, row in df.iterrows():

            reasons = inspect_transcription(row["text"])

            if reasons:

                split_flags.append({
                    "split": split,
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "text": row["text"],
                    "text_length": len(str(row["text"])),
                    "reasons": "; ".join(reasons)
                })

        print("Total samples:", len(df))
        print("Flagged samples:", len(split_flags))

        all_flags.extend(split_flags)

    # -------------------------------------------------
    # Save flagged records
    # -------------------------------------------------

    report_path = output_dir / "suspicious_transcriptions.csv"

    report = pd.DataFrame(
        all_flags,
        columns=[
            "split",
            "sample_id",
            "image_path",
            "text",
            "text_length",
            "reasons"
        ]
    )

    report.to_csv(
        report_path,
        index=False
    )

    # -------------------------------------------------
    # Print summary
    # -------------------------------------------------

    print("\n========== AUDIT SUMMARY ==========")

    print("Total flagged records:", len(report))

    if not report.empty:

        print("\nFlags by split:")

        print(
            report.groupby("split")
            .size()
            .to_string()
        )

        print("\nFlagged samples:")

        print(
            report[
                ["split", "sample_id", "text", "reasons"]
            ].to_string(index=False)
        )

    else:
        print("No suspicious transcription patterns found.")

    print("\nReport saved:", report_path)

    print("\nIMPORTANT:")
    print("Flagged samples require visual inspection.")
    print("No dataset records were modified or deleted.")

    print("\n========== AUDIT COMPLETED ==========")


if __name__ == "__main__":
    main()
