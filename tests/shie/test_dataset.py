
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.shie.dataset import IAMDataset, collate_iam_batch


def main():

    print("=" * 55)
    print("SUTRAYA SHIE DATASET PIPELINE TEST")
    print("=" * 55)

    dataset = IAMDataset("train")

    print(f"Training samples: {len(dataset)}")

    # Inspect one sample.
    sample = dataset[0]

    print("\nSingle sample:")
    print("Image shape:", sample["image"].shape)
    print("Target shape:", sample["target"].shape)
    print("Input length:", sample["input_length"])
    print("Target length:", sample["target_length"])
    print("Text:", sample["text"])
    print("Sample ID:", sample["sample_id"])
    print("Pixel minimum:", sample["image"].min().item())
    print("Pixel maximum:", sample["image"].max().item())

    # Test variable-width batching.
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_iam_batch
    )

    batch = next(iter(loader))

    print("\nBatch information:")
    print("Images:", batch["images"].shape)
    print("Targets:", batch["targets"].shape)
    print("Input lengths:", batch["input_lengths"])
    print("Target lengths:", batch["target_lengths"])
    print("Texts:", batch["texts"])

    assert batch["images"].ndim == 4
    assert batch["images"].shape[0] == 2
    assert batch["images"].shape[1] == 1
    assert batch["images"].shape[2] == 32

    assert batch["images"].shape[-1] % 2 == 0

    assert batch["targets"].ndim == 1

    assert batch["input_lengths"].shape[0] == 2
    assert batch["target_lengths"].shape[0] == 2

    assert torch.isfinite(batch["images"]).all()
    assert batch["images"].min() >= 0
    assert batch["images"].max() <= 1

    print("\nDATASET PIPELINE TEST PASSED!")


if __name__ == "__main__":
    main()
