
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path("datasets/iam")

for split in ["train", "validation", "test"]:

    csv_path = ROOT / "annotations" / f"{split}.csv"

    df = pd.read_csv(csv_path)

    print(f"\n{split.upper()}")
    print("-" * 30)

    print("Samples:", len(df))
    print("Columns:", list(df.columns))

    missing = 0

    for relative_path in df["image_path"]:
        image_path = Path(relative_path)

        if not image_path.is_file():
            missing += 1

    print("Missing images:", missing)

    if len(df) > 0:
        sample = df.iloc[0]

        image_path = Path(sample["image_path"])

        with Image.open(image_path) as image:
            print("Sample image size:", image.size)
            print("Image mode:", image.mode)

        print("Sample transcription:", sample["text"])

    assert missing == 0, f"{split} has missing images!"

print("\nIAM DATASET VERIFICATION COMPLETED!")
