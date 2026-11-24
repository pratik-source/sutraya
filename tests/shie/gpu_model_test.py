
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.shie.model import SHIEModel
from src.shie.dataset import IAMDataset, collate_iam_batch


def main():

    print("=" * 60)
    print("SUTRAYA SHIE REAL IAM MODEL INTEGRATION TEST")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Select computation device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    # --------------------------------------------------
    # 2. Load actual IAM training dataset
    # --------------------------------------------------

    dataset = IAMDataset("train")

    print("Training samples:", len(dataset))
    print("Vocabulary classes:", dataset.num_classes)
    print("CTC blank index:", dataset.blank_idx)

    # --------------------------------------------------
    # 3. Create a batch of real handwriting images
    # --------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_iam_batch
    )

    batch = next(iter(loader))

    images = batch["images"].to(device)
    targets = batch["targets"].to(device)

    input_lengths = batch["input_lengths"]
    target_lengths = batch["target_lengths"]

    print("\nBATCH INFORMATION")
    print("-" * 40)

    print("Images shape:", images.shape)
    print("Targets shape:", targets.shape)
    print("Input lengths:", input_lengths)
    print("Target lengths:", target_lengths)
    print("Original texts:", batch["texts"])

    # --------------------------------------------------
    # 4. Initialize production SHIE model
    # --------------------------------------------------

    model = SHIEModel(
        num_classes=dataset.num_classes
    ).to(device)

    model.train()

    # --------------------------------------------------
    # 5. Forward propagation
    # --------------------------------------------------

    logits = model(images)

    print("\nMODEL OUTPUT")
    print("-" * 40)

    print("Logits shape:", logits.shape)

    # --------------------------------------------------
    # 6. Verify CNN output lengths
    # --------------------------------------------------

    calculated_lengths = model.get_output_lengths(
        batch["images"].shape[-1] *
        torch.ones(
            len(batch["input_lengths"]),
            dtype=torch.long
        )
    )

    # Use each sample's actual unpadded width-derived
    # input length from the dataset.
    expected_lengths = input_lengths

    assert torch.equal(
        calculated_lengths,
        torch.full_like(
            calculated_lengths,
            logits.shape[1]
        )
    ), "Padded batch output length mismatch."

    assert torch.all(
        expected_lengths <= logits.shape[1]
    ), "Sample input length exceeds model output."

    print("Model output time steps:", logits.shape[1])
    print("Sample CTC input lengths:", expected_lengths)
    print("Output length verification: PASSED")

    # --------------------------------------------------
    # 7. Calculate CTC loss
    # --------------------------------------------------

    log_probs = logits.log_softmax(dim=2)

    # CTC expects [Time, Batch, Classes].
    log_probs = log_probs.permute(1, 0, 2)

    ctc_loss_function = nn.CTCLoss(
        blank=dataset.blank_idx,
        zero_infinity=False
    )

    loss = ctc_loss_function(
        log_probs,
        targets,
        input_lengths,
        target_lengths
    )

    print("\nCTC LOSS")
    print("-" * 40)

    print("Loss:", loss.item())

    assert torch.isfinite(loss), "CTC loss is not finite."

    # --------------------------------------------------
    # 8. Backpropagation
    # --------------------------------------------------

    model.zero_grad(set_to_none=True)

    loss.backward()

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert gradients_exist, "Model gradients not found."

    print("\nBackpropagation: SUCCESSFUL")

    # --------------------------------------------------
    # 9. GPU information
    # --------------------------------------------------

    if device.type == "cuda":

        torch.cuda.synchronize()

        print(
            "GPU:",
            torch.cuda.get_device_name(device)
        )

        print(
            "GPU memory allocated:",
            round(
                torch.cuda.memory_allocated() / 1024**2,
                2
            ),
            "MB"
        )

    # --------------------------------------------------
    # 10. Final result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SHIE REAL IAM MODEL INTEGRATION TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
