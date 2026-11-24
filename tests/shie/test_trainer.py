
import torch

from src.shie.config import load_config
from src.shie.data_loader import create_shie_dataloaders
from src.shie.model import SHIEModel
from src.shie.ctc_loss import SHIECTCLoss
from src.shie.trainer import SHIETrainer


def main():

    print("=" * 65)
    print("SUTRAYA SHIE TRAINER TEST")
    print("=" * 65)

    config = load_config()

    # Select device.
    device_setting = config["hardware"]["device"]

    if device_setting == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_setting)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    print("\nDevice:", device)

    # Load real IAM datasets.
    train_loader, validation_loader = (
        create_shie_dataloaders(config)
    )

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(validation_loader.dataset))

    # Initialize model.
    model = SHIEModel(
        num_classes=config["model"]["num_classes"]
    )

    # Initialize CTC loss.
    criterion = SHIECTCLoss(
        blank_index=config["training"]["blank_index"]
    )

    # Initialize AdamW.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    # Initialize trainer.
    trainer = SHIETrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        gradient_clip=config["training"]["gradient_clipping"]
    )

    # -----------------------------------
    # Test 1: Training two batches
    # -----------------------------------

    print("\nTEST 1: TRAINING")

    training_metrics = trainer.train_one_epoch(
        train_loader,
        max_batches=2
    )

    print("Training metrics:", training_metrics)

    assert training_metrics["batches"] == 2
    assert training_metrics["samples"] == 16
    assert torch.isfinite(
        torch.tensor(training_metrics["loss"])
    )

    print("Training test PASSED")

    # -----------------------------------
    # Test 2: Validation two batches
    # -----------------------------------

    print("\nTEST 2: VALIDATION")

    validation_metrics = trainer.validate_one_epoch(
        validation_loader,
        max_batches=2
    )

    print(
        "Validation loss:",
        validation_metrics["loss"]
    )

    print(
        "Validation CER:",
        validation_metrics["cer"]
    )

    print(
        "Validation WER:",
        validation_metrics["wer"]
    )

    print(
        "CER statistics:",
        validation_metrics["cer_statistics"]
    )

    print(
        "WER statistics:",
        validation_metrics["wer_statistics"]
    )

    assert validation_metrics["batches"] == 2
    assert validation_metrics["samples"] == 16

    assert validation_metrics["cer"] is not None
    assert validation_metrics["wer"] is not None

    print("Validation test PASSED")

    # -----------------------------------
    # Test 3: Model evaluation mode
    # -----------------------------------

    assert not trainer.model.training

    print("\nTEST 3: MODEL EVALUATION MODE PASSED")

    # -----------------------------------
    # GPU information
    # -----------------------------------

    if device.type == "cuda":

        print("\nGPU:", torch.cuda.get_device_name(device))

        print(
            "Peak memory:",
            round(
                torch.cuda.max_memory_allocated(device)
                / 1024**2,
                2
            ),
            "MB"
        )

    print("\n" + "=" * 65)
    print("ALL SHIE TRAINER TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
