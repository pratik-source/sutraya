
import torch

from src.shie.config import load_config
from src.shie.data_loader import create_shie_dataloaders
from src.shie.model import SHIEModel
from src.shie.ctc_loss import SHIECTCLoss


def main():

    print("=" * 65)
    print("SUTRAYA SHIE ONE-BATCH TRAINING TEST")
    print("=" * 65)

    # ---------------------------------
    # 1. Configuration and device
    # ---------------------------------

    config = load_config()

    device = torch.device(
        "cuda" if (
            config["hardware"]["device"] == "auto"
            and torch.cuda.is_available()
        ) or config["hardware"]["device"] == "cuda"
        else "cpu"
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    print("\nDevice:", device)

    # ---------------------------------
    # 2. Load real IAM data
    # ---------------------------------

    train_loader, _ = create_shie_dataloaders(config)

    batch = next(iter(train_loader))

    images = batch["images"].to(device)

    targets = batch["targets"]

    input_lengths = batch["input_lengths"]

    target_lengths = batch["target_lengths"]

    print("\nREAL IAM BATCH")
    print("Images:", images.shape)
    print("Targets:", targets.shape)
    print("Input lengths:", input_lengths.tolist())
    print("Target lengths:", target_lengths.tolist())

    # ---------------------------------
    # 3. Initialize model
    # ---------------------------------

    model = SHIEModel(
        num_classes=config["model"]["num_classes"]
    ).to(device)

    model.train()

    # ---------------------------------
    # 4. Initialize CTC loss
    # ---------------------------------

    criterion = SHIECTCLoss(
        blank_index=config["training"]["blank_index"]
    )

    # ---------------------------------
    # 5. Initialize optimizer
    # ---------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # Save a parameter before training.
    parameter_before = (
        next(model.parameters()).detach().clone()
    )

    optimizer.zero_grad(set_to_none=True)

    # ---------------------------------
    # 6. Forward propagation
    # ---------------------------------

    print("\nFORWARD PROPAGATION")

    logits = model(
        images,
        input_lengths=input_lengths
    )

    print("Logits:", logits.shape)

    assert logits.shape[0] == images.shape[0]
    assert logits.shape[1] == images.shape[-1] // 2
    assert logits.shape[2] == config["model"]["num_classes"]

    # ---------------------------------
    # 7. Calculate CTC loss
    # ---------------------------------

    print("\nCTC LOSS")

    loss = criterion(
        logits,
        targets,
        input_lengths,
        target_lengths
    )

    print("Loss:", loss.item())

    assert torch.isfinite(loss).item()
    assert loss.item() > 0

    # ---------------------------------
    # 8. Backpropagation
    # ---------------------------------

    print("\nBACKPROPAGATION")

    loss.backward()

    gradient_count = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            assert torch.isfinite(
                parameter.grad
            ).all().item()

            gradient_count += 1

    assert gradient_count > 0

    print("Parameters with gradients:", gradient_count)

    # ---------------------------------
    # 9. Gradient clipping
    # ---------------------------------

    max_norm = config["training"]["gradient_clipping"]

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm
    )

    print("Gradient norm before clipping:", gradient_norm.item())

    # ---------------------------------
    # 10. Optimizer update
    # ---------------------------------

    optimizer.step()

    parameter_after = next(model.parameters()).detach()

    parameter_changed = not torch.equal(
        parameter_before,
        parameter_after
    )

    assert parameter_changed

    print("\nOPTIMIZER")
    print("AdamW parameter update: SUCCESS")
    print("Model parameters changed:", parameter_changed)

    # ---------------------------------
    # 11. GPU information
    # ---------------------------------

    if device.type == "cuda":

        print("\nGPU INFORMATION")

        print("GPU:", torch.cuda.get_device_name(device))

        print(
            "Allocated memory:",
            round(
                torch.cuda.memory_allocated(device) / 1024**2,
                2
            ),
            "MB"
        )

        print(
            "Peak memory:",
            round(
                torch.cuda.max_memory_allocated(device) / 1024**2,
                2
            ),
            "MB"
        )

    print("\n" + "=" * 65)
    print("SHIE ONE-BATCH TRAINING TEST PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
