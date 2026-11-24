
import torch
import random
import numpy as np

from src.shie.config import load_config
from src.shie.model import SHIEModel
from src.shie.ctc_loss import SHIECTCLoss
from src.shie.data_loader import create_shie_dataloaders
from src.shie.trainer import SHIETrainer
from src.shie.checkpoint import SHIECheckpointManager
from src.shie.history import SHIEHistoryManager


def set_random_seed(seed: int):
    """
    Set random seeds for Python, NumPy and PyTorch.

    This improves reproducibility across SHIE training experiments.
    """

    # Python random generator
    random.seed(seed)

    # NumPy random generator
    np.random.seed(seed)

    # PyTorch CPU random generator
    torch.manual_seed(seed)

    # PyTorch CUDA random generators
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Random seed initialized: {seed}")

def resolve_device(device_config):
    """
    Resolve the training device from configuration.
    """

    if device_config == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device_config == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA device is available."
            )

        return torch.device("cuda")

    return torch.device("cpu")


def train_shie(
    config=None,
    resume=False,
    max_train_batches=None,
    max_validation_batches=None,
    history_path="experiments/shie/training_history.json"
):
    """
    Main SHIE training controller.

    Args:
        config: SHIE configuration dictionary.
        resume: Whether to resume from the last checkpoint.
    """

    print("\n========== SUTRAYA SHIE TRAINING ==========\n")

    # --------------------------------------------------
    # 1. LOAD CONFIGURATION
    # --------------------------------------------------

    if config is None:
        config = load_config()
    
    seed = config["training"].get("seed", 42)
    set_random_seed(seed)

    device = resolve_device(
        config["hardware"]["device"]
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------

    print("\nLoading IAM datasets...")

    train_loader, validation_loader = (
        create_shie_dataloaders(config)
    )

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    # --------------------------------------------------
    # 3. INITIALIZE MODEL
    # --------------------------------------------------

    print("\nInitializing SHIE model...")

    model = SHIEModel(
        num_classes=config["model"]["num_classes"],
        config=config
    ).to(device)

    # --------------------------------------------------
    # 4. INITIALIZE CTC LOSS
    # --------------------------------------------------

    criterion = SHIECTCLoss(
        blank_index=config["training"]["blank_index"]
    )

    # --------------------------------------------------
    # 5. INITIALIZE OPTIMIZER
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    scheduler_config = config["training"]["scheduler"]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_config["factor"],
        patience=scheduler_config["patience"],
        min_lr=scheduler_config["min_lr"],
        threshold=scheduler_config["threshold"]
    )

    # --------------------------------------------------
    # 6. INITIALIZE TRAINER
    # --------------------------------------------------

    trainer = SHIETrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        gradient_clip=config["training"]["gradient_clipping"]
    )

    # --------------------------------------------------
    # 7. INITIALIZE CHECKPOINT MANAGER
    # --------------------------------------------------

    checkpoint_manager = SHIECheckpointManager(
        config["checkpoint"]["directory"]
    )
    
    history_manager = SHIEHistoryManager(history_path)
    
    if not resume:

        if checkpoint_manager.last_checkpoint_path.exists():
            raise FileExistsError(
                "An existing SHIE checkpoint was found. "
                "Use resume=True to continue training, "
                "or explicitly archive the previous experiment "
                "before starting a fresh run."
            )

        history_manager.history["epochs"] = []
        history_manager.save_history()

    best_cer = float("inf")
    start_epoch = 1

    # --------------------------------------------------
    # 8. RESUME TRAINING IF REQUESTED
    # --------------------------------------------------

    if resume:

        print("\nResuming previous SHIE training...")

        metadata = checkpoint_manager.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )

        start_epoch = metadata["epoch"] + 1
        best_cer = metadata["best_cer"]

        print("Resuming from epoch:", start_epoch)
        print("Previous best CER:", best_cer)

    # --------------------------------------------------
    # 9. TRAINING LOOP
    # --------------------------------------------------

    total_epochs = config["training"]["epochs"]

    for epoch in range(start_epoch, total_epochs + 1):

        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        
        train_loader.batch_sampler.set_epoch(epoch)

        print(
            f"Training sampler seed: "
            f"{seed + epoch}"
        )

        print("\n" + "=" * 55)
        print(f"EPOCH {epoch}/{total_epochs}")
        print("=" * 55)

        # TRAINING

        training_metrics = trainer.train_one_epoch(
            train_loader,
            max_batches=max_train_batches
        )

        print("\nTraining Results")
        print("----------------")
        print(f"Loss: {training_metrics['loss']:.4f}")
        print(
            f"Gradient Norm: "
            f"{training_metrics['gradient_norm']:.4f}"
        )
        print("Samples:", training_metrics["samples"])
        print("Batches:", training_metrics["batches"])

        # VALIDATION

        validation_metrics = trainer.validate_one_epoch(
            validation_loader,
            max_batches=max_validation_batches
        )

        print("\nValidation Results")
        print("------------------")
        print(f"Loss: {validation_metrics['loss']:.4f}")
        print(f"CER: {validation_metrics['cer']}")
        print(f"WER: {validation_metrics['wer']}")


        # SAVE BEST MODEL

        # UPDATE LEARNING RATE SCHEDULER

        current_cer = validation_metrics["cer"]

        if current_cer is not None:
            scheduler.step(current_cer)

        next_epoch_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Learning Rate Used: {epoch_learning_rate:.8f}"
        )

        print(
            f"Learning Rate For Next Epoch: {next_epoch_lr:.8f}"
        )

        # SAVE PERSISTENT TRAINING HISTORY

        history_manager.add_epoch(
            epoch=epoch,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            learning_rate=epoch_learning_rate
        )

        if current_cer is not None and current_cer < best_cer:

            best_cer = current_cer

            checkpoint_manager.save_best_model(
                model=model,
                epoch=epoch,
                validation_metrics=validation_metrics
            )

        # SAVE LATEST CHECKPOINT

        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_cer=best_cer,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            scheduler=scheduler
        )

        print("\nEpoch completed:", epoch)
        print("Best CER so far:", best_cer)

    # --------------------------------------------------
    # 10. TRAINING COMPLETED
    # --------------------------------------------------

    print("\n========== SHIE TRAINING COMPLETED ==========")

    print("Total epochs configured:", total_epochs)
    print("Best validation CER:", best_cer)

    print(
        "Best model:",
        checkpoint_manager.best_model_path
    )

    print(
        "Latest checkpoint:",
        checkpoint_manager.last_checkpoint_path
    )

    return history_manager.get_history()


if __name__ == "__main__":
    train_shie(resume=True)
