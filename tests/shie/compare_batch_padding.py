
import random
import math

from src.shie.dataset import IAMDataset
from src.shie.bucket_sampler import WidthBucketBatchSampler


def calculate_padding(widths, batches):
    real_width = 0
    padded_width = 0

    for batch in batches:
        batch_widths = [widths[index] for index in batch]

        maximum_width = max(batch_widths)

        real_width += sum(batch_widths)
        padded_width += maximum_width * len(batch)

    padding_percentage = (
        (padded_width - real_width) / padded_width
    ) * 100

    return padding_percentage, real_width, padded_width


def main():

    print("=" * 60)
    print("SUTRAYA SHIE BATCH PADDING COMPARISON")
    print("=" * 60)

    dataset = IAMDataset(split="train")

    batch_size = 8

    widths = [
        max(
            1,
            round(float(row["width"]) * 32 / float(row["height"]))
        )
        for _, row in dataset.data.iterrows()
    ]

    # ---------------------------------
    # 1. Random batching
    # ---------------------------------

    random_generator = random.Random(42)

    random_indices = list(range(len(dataset)))

    random_generator.shuffle(random_indices)

    random_batches = [
        random_indices[i:i + batch_size]
        for i in range(0, len(random_indices), batch_size)
    ]

    random_padding, random_real, random_padded = (
        calculate_padding(widths, random_batches)
    )

    # ---------------------------------
    # 2. Width bucket batching
    # ---------------------------------

    sampler = WidthBucketBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        bucket_size_multiplier=50,
        shuffle=False,
        drop_last=False
    )

    bucket_batches = list(iter(sampler))

    bucket_padding, bucket_real, bucket_padded = (
        calculate_padding(widths, bucket_batches)
    )

    # ---------------------------------
    # 3. Results
    # ---------------------------------

    print("\nRANDOM BATCHING")
    print("Padding:", round(random_padding, 2), "%")
    print("Real width:", random_real)
    print("Padded width:", random_padded)

    print("\nWIDTH BUCKET BATCHING")
    print("Padding:", round(bucket_padding, 2), "%")
    print("Real width:", bucket_real)
    print("Padded width:", bucket_padded)

    improvement = random_padding - bucket_padding

    print("\nCOMPARISON")
    print("Padding reduction:", round(improvement, 2), "percentage points")

    if random_padding > 0:
        relative_reduction = (
            improvement / random_padding
        ) * 100

        print(
            "Relative padding reduction:",
            round(relative_reduction, 2),
            "%"
        )

    assert len(random_batches) == len(bucket_batches)

    print("\nPadding comparison completed successfully!")


if __name__ == "__main__":
    main()
