
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import load_dataset
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_ID = "test_002658"
SOURCE_INDEX = 2658

CSV_PATH = PROJECT_ROOT / "datasets/iam/annotations/test.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/shie/error_analysis/source_inspection"
)


def main():

    print("\n========== IAM SOURCE INSPECTION ==========\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the exported annotation.
    annotations = pd.read_csv(
        CSV_PATH,
        dtype={"sample_id": str},
        keep_default_na=False
    )

    matches = annotations[
        annotations["sample_id"] == SAMPLE_ID
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one CSV record for {SAMPLE_ID}"
        )

    exported_record = matches.iloc[0]

    exported_image_path = (
        PROJECT_ROOT / exported_record["image_path"]
    )

    print("Exported sample:")
    print("ID:", exported_record["sample_id"])
    print("Text:", repr(exported_record["text"]))
    print("Image:", exported_image_path)

    # 2. Load the original Hugging Face test split.
    print("\nLoading Teklia/IAM-line test split...")

    dataset = load_dataset(
        "Teklia/IAM-line",
        split="test"
    )

    print("Dataset size:", len(dataset))
    print("Inspecting original index:", SOURCE_INDEX)

    if SOURCE_INDEX >= len(dataset):
        raise IndexError("Source index is outside the dataset.")

    source_record = dataset[SOURCE_INDEX]

    source_image = source_record["image"].convert("L")
    source_text = source_record["text"]

    print("\nOriginal Hugging Face sample:")
    print("Index:", SOURCE_INDEX)
    print("Text:", repr(source_text))
    print("Image dimensions:", source_image.size)

    # 3. Load the exported image.
    if not exported_image_path.is_file():
        raise FileNotFoundError(exported_image_path)

    exported_image = Image.open(
        exported_image_path
    ).convert("L")

    print("\nExported image dimensions:", exported_image.size)

    # 4. Compare source and exported images.
    if source_image.size == exported_image.size:

        source_array = np.asarray(
            source_image,
            dtype=np.float32
        )

        exported_array = np.asarray(
            exported_image,
            dtype=np.float32
        )

        mae = np.mean(
            np.abs(source_array - exported_array)
        )

        print("Image mean absolute difference:", round(float(mae), 4))

    else:
        print("WARNING: Image dimensions do not match.")
        print("Image comparison skipped.")

    # 5. Save the original source image for inspection.
    source_output = OUTPUT_DIR / "source_test_002658.png"

    source_image.save(source_output)

    print("\n========== COMPARISON ==========")

    print(
        "Text matches:",
        source_text == exported_record["text"]
    )

    print(
        "Source image saved:",
        source_output
    )

    print("\n================================")


if __name__ == "__main__":
    main()
