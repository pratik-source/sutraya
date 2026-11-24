
import torch

from src.shie.ctc_loss import SHIECTCLoss


def main():

    print("=" * 55)
    print("SUTRAYA SHIE CTC LOSS TEST")
    print("=" * 55)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    criterion = SHIECTCLoss(
        blank_index=0,
        reduction="mean",
        zero_infinity=True
    )

    # Two samples, eight time steps, eighty classes.
    logits = torch.randn(
        2, 8, 80,
        device=device,
        requires_grad=True
    )

    targets = torch.tensor(
        [8, 5, 12, 12, 15, 19],
        dtype=torch.long
    )

    input_lengths = torch.tensor([8, 7])

    target_lengths = torch.tensor([3, 3])

    # Test 1: Forward loss calculation.
    loss = criterion(
        logits,
        targets,
        input_lengths,
        target_lengths
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    print("Test 1: CTC loss calculation PASSED")
    print("Loss:", loss.item())

    # Test 2: Backpropagation.
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    print("Test 2: Backpropagation PASSED")

    # Test 3: Invalid blank target.
    invalid_targets = torch.tensor(
        [8, 0, 12, 12, 15, 19]
    )

    try:
        criterion(
            logits.detach(),
            invalid_targets,
            input_lengths,
            target_lengths
        )
    except ValueError:
        print("Test 3: Blank target validation PASSED")
    else:
        raise AssertionError(
            "Blank target validation failed."
        )

    # Test 4: Invalid target lengths.
    try:
        criterion(
            logits.detach(),
            targets,
            input_lengths,
            torch.tensor([2, 2])
        )
    except ValueError:
        print("Test 4: Target length validation PASSED")
    else:
        raise AssertionError(
            "Target length validation failed."
        )

    # Test 5: Invalid input lengths.
    try:
        criterion(
            logits.detach(),
            targets,
            torch.tensor([9, 7]),
            target_lengths
        )
    except ValueError:
        print("Test 5: Input length validation PASSED")
    else:
        raise AssertionError(
            "Input length validation failed."
        )

    print("=" * 55)
    print("ALL CTC LOSS TESTS PASSED!")
    print("=" * 55)


if __name__ == "__main__":
    main()
