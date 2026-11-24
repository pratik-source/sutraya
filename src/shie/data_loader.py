
from functools import partial

from torch.utils.data import DataLoader

from src.shie.dataset import IAMDataset, collate_iam_batch
from src.shie.bucket_sampler import WidthBucketBatchSampler


def create_shie_dataloaders(config):

    # ----------------------------------------------
    # 1. Read configuration
    # ----------------------------------------------

    training_config = config["training"]
    hardware_config = config["hardware"]
    dataset_config = config["dataset"]

    batch_size = training_config["batch_size"]

    seed = training_config.get("seed", 42)

    num_workers = hardware_config["num_workers"]

    pin_memory = hardware_config["pin_memory"]

    image_height = dataset_config["image_height"]

    image_channels = dataset_config["image_channels"]

    horizontal_stride = dataset_config["horizontal_stride"]

    background_value = dataset_config.get(
        "background_value",
        1.0
    )

    # ----------------------------------------------
    # 2. Create configured datasets
    # ----------------------------------------------

    train_dataset = IAMDataset(
        split="train",
        config=config
    )

    validation_dataset = IAMDataset(
        split="validation",
        config=config
    )

    # ----------------------------------------------
    # 3. Verify dataset settings
    # ----------------------------------------------

    if train_dataset.image_height != image_height:
        raise ValueError(
            "Training dataset image height does not match configuration."
        )

    if validation_dataset.image_height != image_height:
        raise ValueError(
            "Validation dataset image height does not match configuration."
        )

    # ----------------------------------------------
    # 4. Create configured collate function
    # ----------------------------------------------

    collate_fn = partial(
        collate_iam_batch,
        image_height=image_height,
        horizontal_stride=horizontal_stride,
        background_value=background_value,
        image_channels=image_channels,
    )

    # ----------------------------------------------
    # 5. Create training bucket sampler
    # ----------------------------------------------

    train_batch_sampler = WidthBucketBatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        bucket_size_multiplier=50,
        shuffle=True,
        drop_last=False,
        seed=seed,
        image_height=image_height,
    )

    # ----------------------------------------------
    # 6. Create validation bucket sampler
    # ----------------------------------------------

    validation_batch_sampler = WidthBucketBatchSampler(
        dataset=validation_dataset,
        batch_size=batch_size,
        bucket_size_multiplier=50,
        shuffle=False,
        drop_last=False,
        seed=seed,
        image_height=image_height,
    )

    # ----------------------------------------------
    # 7. Training DataLoader
    # ----------------------------------------------

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    # ----------------------------------------------
    # 8. Validation DataLoader
    # ----------------------------------------------

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_sampler=validation_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, validation_loader
