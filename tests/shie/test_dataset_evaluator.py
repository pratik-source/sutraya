
from src.shie.evaluator import SHIEEvaluator


def main():

    print("=" * 55)
    print("SUTRAYA SHIE DATASET EVALUATION TEST")
    print("=" * 55)

    # Test 1: Edit operation statistics.
    result = SHIEEvaluator.edit_statistics(
        "hello",
        "helo"
    )

    assert result["deletions"] == 1
    assert result["insertions"] == 0
    assert result["substitutions"] == 0
    assert result["total_errors"] == 1

    print("Test 1: Edit statistics PASSED")

    # Test 2: Aggregate CER.
    references = ["hello", "world"]
    predictions = ["helo", "word"]

    result = SHIEEvaluator.evaluate_cer_dataset(
        references,
        predictions
    )

    assert result["samples"] == 2
    assert result["reference_units"] == 10
    assert result["total_errors"] == 2
    assert result["error_rate"] == 0.20

    print("Test 2: Aggregate CER PASSED")

    # Test 3: Aggregate WER.
    references = [
        "I love machine learning",
        "SUTRAYA is intelligent"
    ]

    predictions = [
        "I love deep learning",
        "SUTRAYA is intelligent"
    ]

    result = SHIEEvaluator.evaluate_wer_dataset(
        references,
        predictions
    )

    assert result["samples"] == 2
    assert result["reference_units"] == 7
    assert result["total_errors"] == 1

    expected_wer = 1 / 7

    assert abs(result["error_rate"] - expected_wer) < 1e-9

    print("Test 3: Aggregate WER PASSED")

    # Test 4: Dataset with empty reference.
    result = SHIEEvaluator.evaluate_cer_dataset(
        ["", "hello"],
        ["abc", "hello"]
    )

    assert result["empty_reference_samples"] == 1
    assert result["total_errors"] == 3
    assert result["reference_units"] == 5
    assert result["error_rate"] == 0.60

    print("Test 4: Empty reference handling PASSED")

    # Test 5: Entire dataset has empty references.
    result = SHIEEvaluator.evaluate_cer_dataset(
        ["", ""],
        ["", ""]
    )

    assert result["reference_units"] == 0
    assert result["error_rate"] is None

    print("Test 5: Zero denominator handling PASSED")

    # Test 6: Mismatched dataset sizes.
    try:
        SHIEEvaluator.evaluate_cer_dataset(
            ["hello"],
            ["hello", "world"]
        )
    except ValueError:
        print("Test 6: Dataset size validation PASSED")
    else:
        raise AssertionError("Expected ValueError was not raised.")

    print("=" * 55)
    print("ALL DATASET EVALUATION TESTS PASSED!")
    print("=" * 55)


if __name__ == "__main__":
    main()
