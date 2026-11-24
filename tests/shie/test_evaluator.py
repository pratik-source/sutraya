
from src.shie.evaluator import SHIEEvaluator


def main():

    print("=" * 55)
    print("SUTRAYA SHIE LEVENSHTEIN DISTANCE TEST")
    print("=" * 55)

    # Test 1: Identical strings
    result = SHIEEvaluator.levenshtein_distance(
        "hello",
        "hello"
    )

    assert result == 0
    print("Test 1: Identical strings PASSED")

    # Test 2: One deletion
    result = SHIEEvaluator.levenshtein_distance(
        "hello",
        "helo"
    )

    assert result == 1
    print("Test 2: Deletion PASSED")

    # Test 3: One insertion
    result = SHIEEvaluator.levenshtein_distance(
        "cat",
        "cart"
    )

    assert result == 1
    print("Test 3: Insertion PASSED")

    # Test 4: One substitution
    result = SHIEEvaluator.levenshtein_distance(
        "cat",
        "cut"
    )

    assert result == 1
    print("Test 4: Substitution PASSED")

    # Test 5: Multiple edits
    result = SHIEEvaluator.levenshtein_distance(
        "kitten",
        "sitting"
    )

    assert result == 3
    print("Test 5: Multiple edits PASSED")

    # Test 6: Word-level distance
    result = SHIEEvaluator.levenshtein_distance(
        ["I", "love", "AI"],
        ["I", "like", "AI"]
    )

    assert result == 1
    print("Test 6: Word-level distance PASSED")

    # Test 7: Empty reference
    result = SHIEEvaluator.levenshtein_distance(
        "",
        "hello"
    )

    assert result == 5
    print("Test 7: Empty reference PASSED")
    
    
    # Test 8: Character Error Rate
    cer = SHIEEvaluator.calculate_cer(
        "hello",
        "helo"
    )

    assert abs(cer - 0.20) < 1e-9
    print("Test 8: Character Error Rate PASSED")

    # Test 9: Word Error Rate
    wer = SHIEEvaluator.calculate_wer(
        "I love machine learning",
        "I love deep learning"
    )

    assert abs(wer - 0.25) < 1e-9
    print("Test 9: Word Error Rate PASSED")

    # Test 10: Perfect recognition
    cer = SHIEEvaluator.calculate_cer(
        "SUTRAYA",
        "SUTRAYA"
    )

    wer = SHIEEvaluator.calculate_wer(
        "SUTRAYA is intelligent",
        "SUTRAYA is intelligent"
    )

    assert cer == 0.0
    assert wer == 0.0
    print("Test 10: Perfect Recognition PASSED")

    # Test 11: Empty reference
    cer = SHIEEvaluator.calculate_cer("", "hello")
    wer = SHIEEvaluator.calculate_wer("", "hello world")

    assert cer == 1.0
    assert wer == 1.0
    print("Test 11: Empty Reference PASSED")

    # Test 12: Empty prediction
    cer = SHIEEvaluator.calculate_cer("hello", "")
    wer = SHIEEvaluator.calculate_wer("hello world", "")

    assert cer == 1.0
    assert wer == 1.0
    print("Test 12: Empty Prediction PASSED")


    print("=" * 55)
    print("ALL LEVENSHTEIN TESTS PASSED!")
    print("=" * 55)


if __name__ == "__main__":
    main()
