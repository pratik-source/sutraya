
from datasets import load_dataset

print("Connecting to SUTRAYA IAM dataset...")

try:
    dataset = load_dataset(
        "Teklia/IAM-line",
        split="train"
    )

    print("\nDataset loaded successfully!")

    print("Training samples:", len(dataset))
    print("Available columns:", dataset.column_names)

    sample = dataset[0]

    print("\nSample transcription:")
    print(sample["text"])

    print("\nImage information:")
    print("Format:", sample["image"].format)
    print("Dimensions:", sample["image"].size)
    print("Mode:", sample["image"].mode)

    sample["image"].save(
        "datasets/iam/sample_handwriting.png"
    )

    print("\nSample saved successfully!")

except Exception as error:
    print("\nDataset access failed:")
    print(type(error).__name__, error)
