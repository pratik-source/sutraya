
from collections import Counter

from src.shie.dataset import IAMDataset
from src.shie.bucket_sampler import WidthBucketBatchSampler


def main():

    print("=" * 60)
    print("SUTRAYA SHIE WIDTH BUCKET SAMPLER TEST")
    print("=" * 60)

    dataset = IAMDataset(split="train")

    print("\nDataset information")
    print("Training samples:", len(dataset))

    sampler = WidthBucketBatchSampler(
        dataset=dataset,
        batch_size=8,
        bucket_size_multiplier=50,
        shuffle=True,
        drop_last=False
    )

    batches = list(sampler)

    print("\nSampler information")
    print("Batch size:", sampler.batch_size)
    print("Bucket size:", sampler.bucket_size)
    print("Total batches:", len(batches))

    # Verify that all dataset indices are included.
    all_indices = [
        index
        for batch in batches
        for index in batch
    ]

    assert len(all_indices) == len(dataset)

    assert len(set(all_indices)) == len(dataset)

    assert set(all_indices) == set(range(len(dataset)))

    print("\nTest 1: Every sample appears exactly once PASSED")

    # Verify batch sizes.
    for batch in batches:

        assert 0 < len(batch) <= sampler.batch_size

    print("Test 2: Batch sizes are valid PASSED")

    # Verify that batches are width-sorted.
    for batch in batches:

        batch_widths = [
            sampler.widths[index]
            for index in batch
        ]

        assert batch_widths == sorted(batch_widths)

    print("Test 3: Samples are sorted within batches PASSED")

    # Calculate padding statistics.
    total_real_pixels = 0
    total_padded_pixels = 0

    for batch in batches:

        widths = [
            sampler.widths[index]
            for index in batch
        ]

        max_width = max(widths)

        total_real_pixels += sum(widths)

        total_padded_pixels += (
            max_width * len(widths)
        )

    padding_ratio = (
        1 - total_real_pixels / total_padded_pixels
    ) * 100

    print("\nPADDING STATISTICS")
    print("Total batches:", len(batches))
    print("Real image width units:", total_real_pixels)
    print("Padded batch width units:", total_padded_pixels)
    print("Padding percentage:", round(padding_ratio, 2), "%")

    print("\nAll sampler assertions passed!")

    print("=" * 60)
    print("SHIE BUCKET SAMPLER TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
