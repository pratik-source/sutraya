
import json
import math

import cv2
import pandas as pd

from src.shie.config import load_config, resolve_project_path


def count_repeated_characters(text):
    """
    Count adjacent repeated characters.

    Example:
        hello -> 1 (ll)
        book  -> 1 (oo)
        letter -> 2 (tt, ee)
    """

    return sum(
        text[index] == text[index - 1]
        for index in range(1, len(text))
    )


def get_width_category(width):
    if width < 256:
        return "short"

    if width < 512:
        return "medium"

    if width < 1024:
        return "long"

    return "very_long"


def analyze_split(split, csv_path, config, vocabulary):

    print(f"\nAnalyzing {split.upper()} split...")

    df = pd.read_csv(
        resolve_project_path(csv_path),
        keep_default_na=False
    )

    image_height = config["dataset"]["image_height"]
    horizontal_stride = config["dataset"]["horizontal_stride"]

    char_to_idx = vocabulary["char_to_idx"]

    results = []

    for _, row in df.iterrows():

        sample_id = str(row["sample_id"])
        text = str(row["text"])

        image_path = resolve_project_path(
            row["image_path"]
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"Cannot read image: {image_path}"
            )

        original_height, original_width = image.shape

        if original_height <= 0 or original_width <= 0:
            raise ValueError(
                f"Invalid image dimensions: {sample_id}"
            )

        # Use exactly the same width calculation as IAMDataset.
        resized_width = round(
            original_width * image_height / original_height
        )

        resized_width = max(1, resized_width)

        # CNN horizontal output sequence length.
        input_length = (
            resized_width // horizontal_stride
        )

        # Verify that all characters exist in the vocabulary.
        unknown_characters = sorted(
            set(text) - set(char_to_idx)
        )

        if unknown_characters:
            raise ValueError(
                f"Unknown characters in {sample_id}: "
                f"{unknown_characters}"
            )

        target_length = len(text)

        repeated_count = count_repeated_characters(text)

        minimum_ctc_length = (
            target_length + repeated_count
        )

        ctc_slack = (
            input_length - minimum_ctc_length
        )

        ctc_feasible = ctc_slack >= 0

        results.append({

            "split": split,
            "sample_id": sample_id,
            "text": text,

            "original_width": original_width,
            "original_height": original_height,

            "resized_width": resized_width,

            "aspect_ratio": (
                original_width / original_height
            ),

            "input_length": input_length,

            "target_length": target_length,

            "repeated_characters": repeated_count,

            "minimum_ctc_length": minimum_ctc_length,

            "ctc_slack": ctc_slack,

            "ctc_feasible": ctc_feasible,

            "width_category": get_width_category(
                input_length
            )

        })

    return pd.DataFrame(results)


def print_statistics(df, split):

    print("\n" + "=" * 65)
    print(f"{split.upper()} SEQUENCE STATISTICS")
    print("=" * 65)

    print(f"Samples: {len(df)}")

    columns = [
        "original_width",
        "resized_width",
        "input_length",
        "target_length",
        "minimum_ctc_length",
        "ctc_slack"
    ]

    print("\nDistribution:")

    print(
        df[columns].describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        ).round(2).to_string()
    )

    print("\nWidth categories:")

    print(
        df["width_category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCTC feasibility:")

    feasible = int(df["ctc_feasible"].sum())

    print(f"Feasible: {feasible}/{len(df)}")

    print(
        f"Infeasible: {len(df) - feasible}"
    )

    print("\nCTC slack categories:")

    print(
        pd.cut(
            df["ctc_slack"],
            bins=[
                -math.inf,
                -1,
                0,
                5,
                10,
                25,
                50,
                math.inf
            ],
            labels=[
                "Infeasible",
                "Zero slack",
                "1-5",
                "6-10",
                "11-25",
                "26-50",
                "Above 50"
            ]
        ).value_counts(
            sort=False
        ).to_string()
    )


def main():

    print("\n========== SHIE SEQUENCE LENGTH ANALYSIS ==========\n")

    config = load_config()

    vocabulary_path = resolve_project_path(
        config["dataset"]["vocabulary_path"]
    )

    with vocabulary_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        vocabulary = json.load(file)

    output_dir = resolve_project_path(
        "experiments/shie/sequence_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    splits = {
        "train": config["dataset"]["train_csv"],
        "validation": config["dataset"]["validation_csv"]
    }

    all_results = []

    for split, csv_path in splits.items():

        result = analyze_split(
            split=split,
            csv_path=csv_path,
            config=config,
            vocabulary=vocabulary
        )

        print_statistics(result, split)

        result_path = (
            output_dir / f"{split}_sequence_analysis.csv"
        )

        result.to_csv(
            result_path,
            index=False
        )

        print(f"\nDetailed report saved: {result_path}")

        all_results.append(result)

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined_path = (
        output_dir / "combined_sequence_analysis.csv"
    )

    combined.to_csv(
        combined_path,
        index=False
    )

    # Identify samples with the smallest CTC slack.
    difficult = combined.sort_values(
        by="ctc_slack",
        ascending=True
    ).head(30)

    difficult_path = (
        output_dir / "lowest_ctc_slack_samples.csv"
    )

    difficult.to_csv(
        difficult_path,
        index=False
    )

    print("\n========== FINAL SUMMARY ==========")

    print("Total analyzed:", len(combined))

    print(
        "CTC infeasible:",
        int((~combined["ctc_feasible"]).sum())
    )

    print(
        "Minimum CTC slack:",
        combined["ctc_slack"].min()
    )

    print(
        "Median CTC slack:",
        combined["ctc_slack"].median()
    )

    print("\nReports saved in:", output_dir)

    print("\n========== ANALYSIS COMPLETED ==========")


if __name__ == "__main__":
    main()
