
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np


ROOT = Path("datasets/iam/annotations")

SPLITS = ["train", "validation", "test"]

TARGET_HEIGHT = 32

character_counts = Counter()
all_lengths = []
all_widths = []
all_resized_widths = []

total_samples = 0
empty_transcriptions = 0
non_ascii_samples = 0
ctc_infeasible = 0


def count_adjacent_repeats(text):
    return sum(
        text[i] == text[i - 1]
        for i in range(1, len(text))
    )


def main():

    global total_samples
    global empty_transcriptions
    global non_ascii_samples
    global ctc_infeasible

    print("=" * 60)
    print("SUTRAYA IAM DATASET AUDIT")
    print("=" * 60)

    for split in SPLITS:

        csv_path = ROOT / f"{split}.csv"
        df = pd.read_csv(csv_path)

        print(f"\n{split.upper()} DATASET")
        print("-" * 40)

        print("Samples:", len(df))

        split_lengths = []
        split_widths = []

        for _, row in df.iterrows():

            text = str(row["text"])
            width = int(row["width"])
            height = int(row["height"])

            if not text.strip():
                empty_transcriptions += 1
                continue

            total_samples += 1

            character_counts.update(text)

            text_length = len(text)

            resized_width = round(
                width * TARGET_HEIGHT / height
            )

            # Current SHIE CNN reduces width by 2.
            output_length = resized_width // 2

            # CTC needs an additional blank between
            # consecutive repeated target characters.
            required_length = (
                text_length +
                count_adjacent_repeats(text)
            )

            if output_length < required_length:
                ctc_infeasible += 1

            if not text.isascii():
                non_ascii_samples += 1

            all_lengths.append(text_length)
            all_widths.append(width)
            all_resized_widths.append(resized_width)

            split_lengths.append(text_length)
            split_widths.append(width)

        print(
            "Average text length:",
            round(np.mean(split_lengths), 2)
        )

        print(
            "Maximum text length:",
            max(split_lengths)
        )

        print(
            "Average original width:",
            round(np.mean(split_widths), 2)
        )

    print("\n" + "=" * 60)
    print("OVERALL DATASET STATISTICS")
    print("=" * 60)

    print("Valid samples:", total_samples)
    print("Empty transcriptions:", empty_transcriptions)
    print("Non-ASCII samples:", non_ascii_samples)
    print("CTC-infeasible samples:", ctc_infeasible)

    print("\nTRANSCRIPTION LENGTHS")

    for percentile in [25, 50, 75, 90, 95, 99, 100]:
        print(
            f"P{percentile}:",
            round(np.percentile(all_lengths, percentile), 2)
        )

    print("\nRESIZED IMAGE WIDTHS")

    for percentile in [25, 50, 75, 90, 95, 99, 100]:
        print(
            f"P{percentile}:",
            round(np.percentile(all_resized_widths, percentile), 2)
        )

    print("\nCHARACTER VOCABULARY")

    print("Unique characters:", len(character_counts))

    print("\nMost frequent characters:")

    for character, count in character_counts.most_common(30):

        display = repr(character)

        print(f"{display:12} : {count}")

    print("\nALL CHARACTER VALUES")

    print(sorted(character_counts.keys()))

    print("\nDATASET AUDIT COMPLETED")


if __name__ == "__main__":
    main()
