
from pathlib import Path
import pandas as pd

ROOT = Path("datasets/iam/annotations")

SPLITS = ["train", "validation", "test"]

def repeated_characters(text):
    return sum(
        text[i] == text[i - 1]
        for i in range(1, len(text))
    )


def main():

    print("SUTRAYA CTC EDGE CASE INSPECTION")
    print("=" * 65)

    problematic = []

    for split in SPLITS:

        df = pd.read_csv(ROOT / f"{split}.csv")

        for _, row in df.iterrows():

            text = str(row["text"])

            width = int(row["width"])
            height = int(row["height"])

            resized_width = round(
                width * 32 / height
            )

            input_length = resized_width // 2

            required_length = (
                len(text) + repeated_characters(text)
            )

            if input_length < required_length:

                problematic.append({
                    "split": split,
                    "sample_id": row["sample_id"],
                    "text": text,
                    "original_width": width,
                    "original_height": height,
                    "resized_width": resized_width,
                    "ctc_input_length": input_length,
                    "required_length": required_length,
                    "deficit": required_length - input_length
                })

    print("Problematic samples:", len(problematic))

    for sample in problematic:

        print("\n" + "-" * 65)

        for key, value in sample.items():
            print(f"{key}: {value}")

    print("\nINSPECTION COMPLETED")


if __name__ == "__main__":
    main()
