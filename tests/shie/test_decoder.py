
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.shie.decoder import CTCDecoder


def main():

    print("=" * 55)
    print("SUTRAYA SHIE CTC DECODER TEST")
    print("=" * 55)

    # Initialize decoder.
    decoder = CTCDecoder()

    # ------------------------------------------
    # Test 1: Repeated character removal
    # ------------------------------------------

    l = decoder.char_to_idx["l"]

    result = decoder.decode_sequence([
        l, l, l
    ])

    assert result == "l"

    print("Test 1: Repeated character removal PASSED")

    # ------------------------------------------
    # Test 2: Blank removal
    # ------------------------------------------

    h = decoder.char_to_idx["h"]
    e = decoder.char_to_idx["e"]
    o = decoder.char_to_idx["o"]

    result = decoder.decode_sequence([
        h, 0, e, 0, o
    ])

    assert result == "heo"

    print("Test 2: Blank removal PASSED")

    # ------------------------------------------
    # Test 3: Repeated characters separated by blank
    # ------------------------------------------

    result = decoder.decode_sequence([
        l, 0, l
    ])

    assert result == "ll"

    print("Test 3: Blank-separated repeats PASSED")

    # ------------------------------------------
    # Test 4: Complete word decoding
    # ------------------------------------------

    result = decoder.decode_sequence([
        h, h, 0,
        e, e, 0,
        l, l, 0,
        l, o, o
    ])

    assert result == "hello"

    print("Test 4: Complete word decoding PASSED")

    # ------------------------------------------
    # Test 5: Batch logits decoding
    # ------------------------------------------

    batch_logits = torch.full(
        (2, 8, decoder.num_classes),
        fill_value=-10.0
    )

    # First sample: hello
    first_sequence = [
        h, h, 0, e, l, 0, l, o
    ]

    # Second sample: heo
    second_sequence = [
        h, 0, e, 0, o, 0, 0, 0
    ]

    for time, token in enumerate(first_sequence):
        batch_logits[0, time, token] = 10.0

    for time, token in enumerate(second_sequence):
        batch_logits[1, time, token] = 10.0

    results = decoder.decode_batch(
        batch_logits,
        input_lengths=torch.tensor([8, 5])
    )

    assert results == ["hello", "heo"]

    print("Test 5: Batch decoding PASSED")


    # ------------------------------------------
    # Final result
    # ------------------------------------------

    print("\nDecoded results:", results)

    print("\n" + "=" * 55)
    print("ALL CTC DECODER TESTS PASSED!")
    print("=" * 55)


if __name__ == "__main__":
    main()
