
import torch
from torch.utils.data import Sampler


class WidthBucketBatchSampler(Sampler):

    def __init__(
        self,
        dataset,
        batch_size,
        bucket_size_multiplier=50,
        shuffle=True,
        drop_last=False,
        seed=42,
        image_height=None,
    ):

        # ------------------------------------------
        # 1. Validate parameters
        # ------------------------------------------

        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size must be a positive integer."
            )

        if (
            not isinstance(bucket_size_multiplier, int)
            or isinstance(bucket_size_multiplier, bool)
            or bucket_size_multiplier <= 0
        ):
            raise ValueError(
                "bucket_size_multiplier must be a positive integer."
            )

        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(
                "seed must be a non-negative integer."
            )

        if not hasattr(dataset, "data"):
            raise ValueError(
                "Dataset must expose its annotation DataFrame as 'data'."
            )

        required_columns = {"width", "height"}

        if not required_columns.issubset(dataset.data.columns):
            raise ValueError(
                "Dataset annotations must contain width and height."
            )

        # ------------------------------------------
        # 2. Obtain configured image height
        # ------------------------------------------

        if image_height is None:
            image_height = getattr(dataset, "image_height", 32)

        if (
            not isinstance(image_height, int)
            or isinstance(image_height, bool)
            or image_height <= 0
        ):
            raise ValueError(
                "image_height must be a positive integer."
            )

        self.image_height = image_height

        # ------------------------------------------
        # 3. Store sampler settings
        # ------------------------------------------

        self.dataset_size = len(dataset)

        self.batch_size = batch_size

        self.bucket_size = (
            batch_size * bucket_size_multiplier
        )

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.seed = seed
        self.epoch = 0

        # ------------------------------------------
        # 4. Calculate resized image widths
        # ------------------------------------------

        self.widths = []

        for _, row in dataset.data.iterrows():

            original_width = float(row["width"])
            original_height = float(row["height"])

            if original_width <= 0 or original_height <= 0:
                raise ValueError(
                    "Image width and height must be positive."
                )

            resized_width = round(
                original_width
                * self.image_height
                / original_height
            )

            self.widths.append(
                max(1, resized_width)
            )

        # ------------------------------------------
        # 5. Verify annotation count
        # ------------------------------------------

        if len(self.widths) != self.dataset_size:
            raise ValueError(
                "Width information does not match dataset size."
            )

    # ----------------------------------------------
    # Set epoch for reproducible shuffling
    # ----------------------------------------------

    def set_epoch(self, epoch):

        if (
            not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or epoch < 0
        ):
            raise ValueError(
                "epoch must be a non-negative integer."
            )

        self.epoch = epoch

    # ----------------------------------------------
    # Generate batches
    # ----------------------------------------------

    def __iter__(self):

        generator = torch.Generator()

        generator.manual_seed(
            self.seed + self.epoch
        )

        # Generate sample indices.
        if self.shuffle:

            indices = torch.randperm(
                self.dataset_size,
                generator=generator
            ).tolist()

        else:

            indices = list(
                range(self.dataset_size)
            )

        batches = []

        # ------------------------------------------
        # Divide indices into width buckets
        # ------------------------------------------

        for start in range(
            0,
            self.dataset_size,
            self.bucket_size
        ):

            bucket_indices = indices[
                start:start + self.bucket_size
            ]

            # Sort samples by resized image width.
            bucket_indices.sort(
                key=lambda index: self.widths[index]
            )

            # --------------------------------------
            # Create batches
            # --------------------------------------

            for batch_start in range(
                0,
                len(bucket_indices),
                self.batch_size
            ):

                batch = bucket_indices[
                    batch_start:batch_start + self.batch_size
                ]

                if (
                    len(batch) == self.batch_size
                    or not self.drop_last
                ):

                    batches.append(batch)

        # ------------------------------------------
        # Shuffle batch order
        # ------------------------------------------

        if self.shuffle:

            permutation = torch.randperm(
                len(batches),
                generator=generator
            ).tolist()

            batches = [
                batches[index]
                for index in permutation
            ]

        yield from batches

    # ----------------------------------------------
    # Number of batches
    # ----------------------------------------------

    def __len__(self):

        if not self.drop_last:

            return (
                self.dataset_size + self.batch_size - 1
            ) // self.batch_size

        total_batches = 0

        for start in range(
            0,
            self.dataset_size,
            self.bucket_size
        ):

            bucket_length = min(
                self.bucket_size,
                self.dataset_size - start
            )

            total_batches += (
                bucket_length // self.batch_size
            )

        return total_batches
