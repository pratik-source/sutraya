
import torch

from src.shie.config import load_config
from src.shie.data_loader import create_shie_dataloaders


def main():

    print("=" * 60)
    print("SUTRAYA SHIE BUCKETED DATALOADER TEST")
    print("=" * 60)

    config = load_config()

    train_loader, validation_loader = (
        create_shie_dataloaders(config)
    )

    print("\nDATASET INFORMATION")

    print("Training samples:", len(train_loader.dataset))

    print("Validation samples:", len(validation_loader.dataset))

    print("Training batches:", len(train_loader))

    print("Validation batches:", len(validation_loader))

    # Verify batch counts.
    expected_train_batches = (
        len(train_loader.dataset)
        + config["training"]["batch_size"] - 1
    ) // config["training"]["batch_size"]

    assert len(train_loader) == expected_train_batches

    # Inspect training batch.
    train_batch = next(iter(train_loader))

    print("\nTRAINING BATCH")

    print("Images:", train_batch["images"].shape)

    print("Targets:", train_batch["targets"].shape)

    print("Input lengths:", train_batch["input_lengths"])

    print("Target lengths:", train_batch["target_lengths"])

    # Inspect validation batch.
    validation_batch = next(iter(validation_loader))

    print("\nVALIDATION BATCH")

    print("Images:", validation_batch["images"].shape)

    print("Input lengths:", validation_batch["input_lengths"])

    # Shape validation.
    for batch in [train_batch, validation_batch]:

        images = batch["images"]

        assert images.ndim == 4

        assert images.shape[1] == 1

        assert images.shape[2] == 32

        assert images.shape[-1] % 2 == 0

        assert torch.isfinite(images).all()

        assert torch.all(
            batch["input_lengths"] <= images.shape[-1] // 2
        )

    print("\nAll DataLoader assertions passed!")

    print("=" * 60)

    print("SHIE BUCKETED DATALOADER TEST PASSED!")

    print("=" * 60)


if __name__ == "__main__":
    main()
