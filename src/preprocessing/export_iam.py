
from pathlib import Path
import csv

from datasets import load_dataset
from tqdm import tqdm


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "iam"

IMAGE_ROOT = DATASET_ROOT / "raw" / "images"
ANNOTATION_ROOT = DATASET_ROOT / "annotations"

SPLITS = ["train", "validation", "test"]


def export_split(dataset, split):

    print(f"\nExporting {split} split...")

    image_dir = IMAGE_ROOT / split
    image_dir.mkdir(parents=True, exist_ok=True)

    ANNOTATION_ROOT.mkdir(parents=True, exist_ok=True)

    csv_path = ANNOTATION_ROOT / f"{split}.csv"

    exported = 0
    skipped = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "image_path",
                "text",
                "split",
                "width",
                "height"
            ]
        )

        writer.writeheader()

        for index, sample in enumerate(
            tqdm(dataset, desc=split)
        ):

            image = sample["image"]
            text = sample["text"]

            if image is None or not isinstance(text, str):
                skipped += 1
                continue

            text = text.strip()

            if not text:
                skipped += 1
                continue

            sample_id = f"{split}_{index:06d}"

            image_path = image_dir / f"{sample_id}.jpg"

            image = image.convert("L")

            width, height = image.size

            image.save(
                image_path,
                format="JPEG",
                quality=95
            )

            relative_path = image_path.relative_to(
                PROJECT_ROOT
            ).as_posix()

            writer.writerow({
                "sample_id": sample_id,
                "image_path": relative_path,
                "text": text,
                "split": split,
                "width": width,
                "height": height
            })

            exported += 1

    print(f"Exported: {exported}")
    print(f"Skipped: {skipped}")
    print(f"Annotation file: {csv_path}")

    return exported, skipped


def main():

    print("SUTRAYA IAM Dataset Exporter")
    print("=" * 40)

    total_exported = 0

    for split in SPLITS:

        print(f"\nLoading {split} dataset...")

        dataset = load_dataset(
            "Teklia/IAM-line",
            split=split
        )

        exported, skipped = export_split(
            dataset,
            split
        )

        total_exported += exported

        del dataset

    print("\n" + "=" * 40)
    print("DATASET EXPORT COMPLETED")
    print("Total exported:", total_exported)
    print("=" * 40)


if __name__ == "__main__":
    main()
