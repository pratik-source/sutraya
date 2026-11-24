
from pathlib import Path
import ast
import math
import textwrap

import cv2
import pandas as pd
import matplotlib.pyplot as plt

from src.shie.config import load_config, resolve_project_path


# Number of samples displayed on each contact sheet.
SAMPLES_PER_SHEET = 12

# Layout: 3 columns x 4 rows.
COLUMNS = 3
ROWS = 4


def load_transcriptions(config):
    """
    Load annotations from all three splits and combine them.
    """

    dataset_config = config["dataset"]

    csv_paths = {
        "train": dataset_config["train_csv"],
        "validation": dataset_config["validation_csv"],
        "test": dataset_config["test_csv"]
    }

    frames = []

    for split, configured_path in csv_paths.items():

        csv_path = resolve_project_path(configured_path)

        df = pd.read_csv(
            csv_path,
            dtype={"sample_id": str},
            keep_default_na=False
        )

        df["split"] = split

        frames.append(
            df[["sample_id", "split", "text"]]
        )

    annotations = pd.concat(
        frames,
        ignore_index=True
    )

    if annotations.duplicated(
        subset=["sample_id", "split"]
    ).any():
        raise ValueError(
            "Duplicate sample IDs found in annotations."
        )

    return annotations


def parse_issues(value):
    """
    Convert the issues CSV representation into a readable list.
    """

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return parsed

    except (ValueError, SyntaxError):
        pass

    return [str(value)]


def create_contact_sheet(samples, sheet_number, output_dir):

    fig, axes = plt.subplots(
        ROWS,
        COLUMNS,
        figsize=(19, 14)
    )

    axes = axes.flatten()

    for index, (_, row) in enumerate(samples.iterrows()):

        ax = axes[index]

        image_path = resolve_project_path(
            row["image_path"]
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            ax.text(
                0.5,
                0.5,
                "IMAGE COULD NOT BE READ",
                ha="center",
                va="center",
                color="red"
            )

            ax.axis("off")
            continue

        # Display original image without modifying its aspect ratio.
        ax.imshow(
            image,
            cmap="gray",
            aspect="auto",
            vmin=0,
            vmax=255
        )

        ax.set_xticks([])
        ax.set_yticks([])

        sample_id = row["sample_id"]
        split = row["split"]

        width = int(row["actual_width"])
        height = int(row["actual_height"])

        ratio = float(row["aspect_ratio"])

        text = str(row["text"])

        wrapped_text = textwrap.fill(
            text,
            width=65
        )

        issues = ", ".join(
            parse_issues(row["issues"])
        )

        title = (
            f"{sample_id} | {split.upper()}\n"
            f"Size: {width} x {height} | "
            f"Aspect ratio: {ratio:.2f}\n"
            f"Issue: {issues}"
        )

        ax.set_title(
            title,
            fontsize=8,
            fontweight="bold",
            loc="left"
        )

        ax.text(
            0.0,
            -0.20,
            f"TEXT: {wrapped_text}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            wrap=True
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)

    # Hide unused subplot cells.
    for index in range(len(samples), len(axes)):
        axes[index].axis("off")

    fig.suptitle(
        f"SUTRAYA SHIE - FLAGGED IMAGE INSPECTION "
        f"(Sheet {sheet_number})",
        fontsize=16,
        fontweight="bold"
    )

    plt.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.92,
        bottom=0.08,
        hspace=0.75,
        wspace=0.15
    )

    output_path = (
        output_dir
        / f"contact_sheet_{sheet_number:02d}.png"
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def main():

    print("\n========== SHIE FLAGGED IMAGE VISUALIZATION ==========\n")

    config = load_config()

    # --------------------------------------------------
    # 1. Locate audit report
    # --------------------------------------------------

    issues_path = resolve_project_path(
        "experiments/shie/dataset_audit/image_integrity_issues.csv"
    )

    if not issues_path.is_file():
        raise FileNotFoundError(
            f"Image integrity report not found: {issues_path}"
        )

    issues_df = pd.read_csv(
        issues_path,
        dtype={"sample_id": str},
        keep_default_na=False
    )

    if issues_df.empty:
        print("No flagged images to visualize.")
        return

    print("Flagged images:", len(issues_df))

    # --------------------------------------------------
    # 2. Load original transcriptions
    # --------------------------------------------------

    print("\nLoading original annotations...")

    annotations = load_transcriptions(config)

    merged = issues_df.merge(
        annotations,
        on=["sample_id", "split"],
        how="left",
        validate="one_to_one"
    )

    if merged["text"].isna().any():
        missing = merged[
            merged["text"].isna()
        ]["sample_id"].tolist()

        raise ValueError(
            f"Missing transcriptions for: {missing}"
        )

    print("Successfully matched all transcriptions.")

    # --------------------------------------------------
    # 3. Sort and prepare output
    # --------------------------------------------------

    merged = merged.sort_values(
        by=["split", "aspect_ratio"],
        ascending=[True, False]
    ).reset_index(drop=True)

    output_dir = resolve_project_path(
        "experiments/shie/dataset_audit/contact_sheets"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save a complete inspection manifest.
    manifest_path = output_dir / "inspection_manifest.csv"

    merged[
        [
            "split",
            "sample_id",
            "image_path",
            "text",
            "actual_width",
            "actual_height",
            "aspect_ratio",
            "foreground_ratio",
            "issues"
        ]
    ].to_csv(
        manifest_path,
        index=False
    )

    # --------------------------------------------------
    # 4. Generate contact sheets
    # --------------------------------------------------

    total_samples = len(merged)

    total_sheets = math.ceil(
        total_samples / SAMPLES_PER_SHEET
    )

    print("\nGenerating contact sheets...")

    for sheet_index in range(total_sheets):

        start = (
            sheet_index * SAMPLES_PER_SHEET
        )

        end = min(
            start + SAMPLES_PER_SHEET,
            total_samples
        )

        samples = merged.iloc[start:end]

        create_contact_sheet(
            samples=samples,
            sheet_number=sheet_index + 1,
            output_dir=output_dir
        )

    # --------------------------------------------------
    # 5. Final summary
    # --------------------------------------------------

    print("\n========== VISUALIZATION COMPLETED ==========")

    print("Total flagged images:", total_samples)
    print("Contact sheets:", total_sheets)

    print("Output directory:", output_dir)

    print("Inspection manifest:", manifest_path)

    print("\nNo images or annotations were modified.")

    print("\n==============================================")


if __name__ == "__main__":
    main()
