
import pandas as pd

from src.shie.bucket_sampler import WidthBucketBatchSampler


class DummyDataset:

    def __init__(self, size=100):

        self.data = pd.DataFrame({
            "width": [
                100 + (i * 17) % 500
                for i in range(size)
            ],
            "height": [
                32
                for _ in range(size)
            ]
        })

    def __len__(self):
        return len(self.data)


def collect_batches(sampler):

    return [
        list(batch)
        for batch in sampler
    ]


def test_same_seed_and_epoch_produce_same_batches():

    dataset = DummyDataset()

    sampler_a = WidthBucketBatchSampler(
        dataset, batch_size=8, seed=42
    )

    sampler_b = WidthBucketBatchSampler(
        dataset, batch_size=8, seed=42
    )

    sampler_a.set_epoch(1)
    sampler_b.set_epoch(1)

    assert collect_batches(sampler_a) == collect_batches(sampler_b)


def test_different_epochs_produce_different_batches():

    dataset = DummyDataset()

    sampler = WidthBucketBatchSampler(
        dataset, batch_size=8, seed=42
    )

    sampler.set_epoch(1)
    epoch_one = collect_batches(sampler)

    sampler.set_epoch(2)
    epoch_two = collect_batches(sampler)

    assert epoch_one != epoch_two


def test_every_sample_appears_exactly_once():

    dataset = DummyDataset()

    sampler = WidthBucketBatchSampler(
        dataset, batch_size=8, seed=42
    )

    batches = collect_batches(sampler)

    all_indices = [
        index
        for batch in batches
        for index in batch
    ]

    assert len(all_indices) == len(dataset)
    assert sorted(all_indices) == list(range(len(dataset)))


def test_repeated_iteration_of_same_epoch_is_identical():

    dataset = DummyDataset()

    sampler = WidthBucketBatchSampler(
        dataset, batch_size=8, seed=42
    )

    sampler.set_epoch(5)

    first_run = collect_batches(sampler)
    second_run = collect_batches(sampler)

    assert first_run == second_run


if __name__ == "__main__":

    test_same_seed_and_epoch_produce_same_batches()
    print("TEST 1 PASSED: Same seed and epoch")

    test_different_epochs_produce_different_batches()
    print("TEST 2 PASSED: Different epochs")

    test_every_sample_appears_exactly_once()
    print("TEST 3 PASSED: Every sample appears exactly once")

    test_repeated_iteration_of_same_epoch_is_identical()
    print("TEST 4 PASSED: Repeated iteration")

    print("\nALL SAMPLER REPRODUCIBILITY TESTS PASSED!")
