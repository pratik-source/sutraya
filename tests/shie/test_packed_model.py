
import torch

from src.shie.config import load_config
from src.shie.model import SHIEModel


def main():

    print("=" * 60)
    print("SUTRAYA SHIE PACKED MODEL TEST")
    print("=" * 60)

    config = load_config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    num_classes = config["model"]["num_classes"]

    model = SHIEModel(
        num_classes=num_classes
    ).to(device)

    model.train()

    # Simulate a batch of padded handwriting images.
    batch_size = 4
    image_width = 400

    images = torch.rand(
        batch_size,
        1,
        32,
        image_width,
        device=device
    )

    # Deliberately unsorted sequence lengths.
    input_lengths = torch.tensor([
        150,
        100,
        180,
        120
    ])

    print("\nINPUT INFORMATION")
    print("Images:", images.shape)
    print("Input lengths:", input_lengths.tolist())

    # ---------------------------------
    # Test 1: Packed forward pass
    # ---------------------------------

    logits = model(
        images,
        input_lengths=input_lengths
    )

    print("\nPACKED FORWARD PASS")
    print("Logits:", logits.shape)

    assert logits.shape == (
        batch_size,
        image_width // 2,
        num_classes
    )

    print("Test 1: Packed forward pass PASSED")

    # ---------------------------------
    # Test 2: Output lengths
    # ---------------------------------

    output_lengths = model.get_output_lengths(
        torch.tensor([300, 200, 360, 240])
    )

    assert torch.equal(
        output_lengths,
        input_lengths
    )

    print("Test 2: Output length calculation PASSED")

    # ---------------------------------
    # Test 3: Backpropagation
    # ---------------------------------

    loss = logits.square().mean()

    loss.backward()

    print("\nBACKPROPAGATION")
    print("Test loss:", loss.item())

    assert torch.isfinite(loss)

    gradients_found = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            assert torch.isfinite(
                parameter.grad
            ).all()

            gradients_found += 1

    assert gradients_found > 0

    print("Parameters with gradients:", gradients_found)

    print("Test 3: Backpropagation PASSED")

    # ---------------------------------
    # Test 4: Without packed sequences
    # ---------------------------------

    model.zero_grad(set_to_none=True)

    normal_logits = model(images)

    assert normal_logits.shape == logits.shape

    print("Test 4: Standard forward compatibility PASSED")

    # ---------------------------------
    # Test 5: GPU memory
    # ---------------------------------

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(device)
        )

        print(
            "Allocated GPU memory:",
            round(
                torch.cuda.memory_allocated(device) / 1024**2,
                2
            ),
            "MB"
        )

    print("\n" + "=" * 60)
    print("ALL PACKED MODEL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()

