
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "datasets/iam/annotations/train.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    print("=" * 55)
    print("SUTRAYA SHIE IMAGE INSPECTION")
    print("=" * 55)

    # Inspect three training images.
    for index in [0, 100, 500]:
        row = df.iloc[index]

        image_path = ROOT / row["image_path"]

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(image_path)

        print(f"\nSample: {row['sample_id']}")
        print(f"Text: {row['text']}")
        print(f"Shape: {image.shape}")
        print(f"Data type: {image.dtype}")
        print(f"Minimum pixel: {image.min()}")
        print(f"Maximum pixel: {image.max()}")
        print(f"Mean pixel: {image.mean():.2f}")
        print(f"Median pixel: {np.median(image):.2f}")

        # Inspect resized image.
        height, width = image.shape

        new_width = round(width * 32 / height)

        resized = cv2.resize(
            image,
            (new_width, 32),
            interpolation=cv2.INTER_AREA
        )

        print(f"Resized shape: {resized.shape}")
        print(f"Resized mean: {resized.mean():.2f}")


if __name__ == "__main__":
    main()
