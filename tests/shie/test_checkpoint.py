
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from src.shie.checkpoint import SHIECheckpointManager


def run_checkpoint_tests():

    print("\n========== SHIE CHECKPOINT TESTS ==========\n")

    test_dir = Path("tests/temp_shie_checkpoints")

    if test_dir.exists():
        shutil.rmtree(test_dir)

    manager = SHIECheckpointManager(test_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # TEST 1: Create model, optimizer and scheduler
    # --------------------------------------------------

    model = nn.Linear(10, 5).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.5
    )

    # Perform one optimizer update to initialize
    # AdamW's internal optimizer state.

    dummy_input = torch.randn(4, 10).to(device)
    dummy_target = torch.randn(4, 5).to(device)

    criterion = nn.MSELoss()

    optimizer.zero_grad()

    output = model(dummy_input)
    loss = criterion(output, dummy_target)

    loss.backward()
    optimizer.step()
    scheduler.step()

    print("\nTEST 1: Model and optimizer initialized")

    assert len(optimizer.state) > 0

    # --------------------------------------------------
    # TEST 2: Save training checkpoint
    # --------------------------------------------------

    training_metrics = {
        "loss": 2.45,
        "gradient_norm": 1.82,
        "samples": 100,
        "batches": 13
    }

    validation_metrics = {
        "loss": 2.12,
        "cer": 0.45,
        "wer": 0.62
    }

    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=5,
        best_cer=0.45,
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        scheduler=scheduler
    )

    assert manager.last_checkpoint_path.exists()

    print("TEST 2: Training checkpoint saved")

    # --------------------------------------------------
    # TEST 3: Modify model and restore checkpoint
    # --------------------------------------------------

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    restored_model = nn.Linear(10, 5).to(device)

    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=0.001
    )

    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer,
        step_size=2,
        gamma=0.5
    )

    metadata = manager.load_checkpoint(
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        device=device
    )

    for name, parameter in restored_model.named_parameters():

        assert torch.equal(
            parameter,
            original_parameters[name]
        ), f"Parameter restoration failed: {name}"

    print("TEST 3: Model parameters restored correctly")

    # --------------------------------------------------
    # TEST 4: Verify optimizer and scheduler restoration
    # --------------------------------------------------

    assert len(restored_optimizer.state) == len(optimizer.state)

    assert (
        restored_optimizer.param_groups[0]["lr"]
        == optimizer.param_groups[0]["lr"]
    )

    assert (
        restored_scheduler.state_dict()
        == scheduler.state_dict()
    )

    print("TEST 4: Optimizer and scheduler restored")

    # --------------------------------------------------
    # TEST 5: Verify checkpoint metadata
    # --------------------------------------------------

    assert metadata["epoch"] == 5
    assert metadata["best_cer"] == 0.45

    assert (
        metadata["training_metrics"]["loss"]
        == 2.45
    )

    assert (
        metadata["validation_metrics"]["cer"]
        == 0.45
    )

    print("TEST 5: Checkpoint metadata restored")

    # --------------------------------------------------
    # TEST 6: Save and inspect best model
    # --------------------------------------------------

    manager.save_best_model(
        model=restored_model,
        epoch=5,
        validation_metrics=validation_metrics
    )

    assert manager.best_model_path.exists()

    best_checkpoint = torch.load(
        manager.best_model_path,
        map_location=device,
        weights_only=True
    )

    assert best_checkpoint["epoch"] == 5

    assert best_checkpoint["validation_cer"] == 0.45

    assert "optimizer_state_dict" not in best_checkpoint

    for name, parameter in restored_model.named_parameters():

        assert torch.equal(
            parameter,
            best_checkpoint["model_state_dict"][name]
        )

    print("TEST 6: Best model saved correctly")

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    shutil.rmtree(test_dir)

    print("\n============================================")
    print("ALL SHIE CHECKPOINT TESTS PASSED!")
    print("============================================")


if __name__ == "__main__":
    run_checkpoint_tests()
