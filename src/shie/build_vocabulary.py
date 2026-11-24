
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = ROOT / "datasets/iam/annotations/train.csv"
VAL_CSV = ROOT / "datasets/iam/annotations/validation.csv"
TEST_CSV = ROOT / "datasets/iam/annotations/test.csv"

OUTPUT_DIR = ROOT / "configs"
OUTPUT_FILE = OUTPUT_DIR / "shie_vocabulary.json"


def main():
    print("=" * 55)
    print("SUTRAYA SHIE VOCABULARY BUILDER")
    print("=" * 55)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Build vocabulary using training data only.
    characters = sorted(
        set("".join(train_df["text"].astype(str)))
    )

    char_to_idx = {
        char: index + 1
        for index, char in enumerate(characters)
    }

    idx_to_char = {
        str(index): char
        for char, index in char_to_idx.items()
    }

    # CTC blank token is reserved at index 0.
    blank_idx = 0

    train_chars = set(characters)
    val_chars = set("".join(val_df["text"].astype(str)))
    test_chars = set("".join(test_df["text"].astype(str)))

    unseen_val = sorted(val_chars - train_chars)
    unseen_test = sorted(test_chars - train_chars)

    print(f"Training characters: {len(characters)}")
    print(f"Total output classes: {len(characters) + 1}")
    print(f"CTC blank index: {blank_idx}")
    print(f"Unseen validation characters: {unseen_val}")
    print(f"Unseen test characters: {unseen_test}")

    if unseen_val or unseen_test:
        raise ValueError(
            "Unseen characters found. Vocabulary needs investigation."
        )

    vocabulary = {
        "blank_token": "<BLANK>",
        "blank_idx": blank_idx,
        "num_classes": len(characters) + 1,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            vocabulary,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"\nVocabulary saved to: {OUTPUT_FILE}")
    print("\nVOCABULARY BUILD COMPLETED")


if __name__ == "__main__":
    main()
