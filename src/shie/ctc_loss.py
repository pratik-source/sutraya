import torch
import torch.nn as nn
import torch.nn.functional as F


class SHIECTCLoss(nn.Module):
    """
    Validated CTC loss wrapper for SHIE.

    Expected model output:
        logits: [batch, time, classes]

    Expected targets:
        One concatenated 1D tensor containing all target tokens.

    Sequence lengths are accepted as integer tensors on CPU or GPU.
    They are converted to CPU int64 before calling PyTorch CTCLoss.
    """

    def __init__(
        self,
        blank_index=0,
        reduction="mean",
        zero_infinity=True
    ):
        super().__init__()

        if isinstance(blank_index, bool) or not isinstance(blank_index, int):
            raise TypeError("blank_index must be an integer.")

        if blank_index < 0:
            raise ValueError("Blank index cannot be negative.")

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "reduction must be one of: 'none', 'mean', or 'sum'."
            )

        if not isinstance(zero_infinity, bool):
            raise TypeError("zero_infinity must be a boolean.")

        self.blank_index = blank_index

        self.ctc_loss = nn.CTCLoss(
            blank=blank_index,
            reduction=reduction,
            zero_infinity=zero_infinity
        )

    @staticmethod
    def _validate_integer_tensor(tensor, name):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")

        if tensor.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8
        }:
            raise TypeError(f"{name} must have an integer dtype.")

    @staticmethod
    def _minimum_ctc_timesteps(sample_targets):
        """
        CTC needs one extra timestep for every adjacent repeated token.

        Example:
            A B B C requires at least 5 timesteps:
            A, B, blank, B, C
        """
        target_length = sample_targets.numel()

        if target_length <= 1:
            return target_length

        repeated_tokens = (
            sample_targets[1:] == sample_targets[:-1]
        ).sum().item()

        return target_length + repeated_tokens

    def forward(
        self,
        logits,
        targets,
        input_lengths,
        target_lengths
    ):
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor.")

        if logits.ndim != 3:
            raise ValueError(
                "Expected logits shape [Batch, Time, Classes]."
            )

        batch_size, time_steps, num_classes = logits.shape

        if batch_size <= 0:
            raise ValueError("Logits batch size must be greater than zero.")

        if time_steps <= 0:
            raise ValueError("Logits must contain at least one timestep.")

        if num_classes <= 1:
            raise ValueError("Logits must contain at least two classes.")

        if self.blank_index >= num_classes:
            raise ValueError(
                "blank_index must be smaller than the number of classes."
            )

        if not logits.is_floating_point():
            raise TypeError("logits must have a floating-point dtype.")

        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("logits contain NaN or infinite values.")

        self._validate_integer_tensor(targets, "targets")
        self._validate_integer_tensor(input_lengths, "input_lengths")
        self._validate_integer_tensor(target_lengths, "target_lengths")

        if targets.ndim != 1:
            raise ValueError(
                "Targets must be a concatenated 1D tensor."
            )

        if input_lengths.ndim != 1:
            raise ValueError("input_lengths must be a 1D tensor.")

        if target_lengths.ndim != 1:
            raise ValueError("target_lengths must be a 1D tensor.")

        if input_lengths.numel() != batch_size:
            raise ValueError(
                "Input lengths must match batch size."
            )

        if target_lengths.numel() != batch_size:
            raise ValueError(
                "Target lengths must match batch size."
            )

        # Move lengths to CPU int64 before inspecting and passing to CTCLoss.
        input_lengths_cpu = input_lengths.detach().to(
            device="cpu",
            dtype=torch.long
        )
        target_lengths_cpu = target_lengths.detach().to(
            device="cpu",
            dtype=torch.long
        )

        if torch.any(input_lengths_cpu <= 0).item():
            raise ValueError("Every input length must be greater than zero.")

        if torch.any(input_lengths_cpu > time_steps).item():
            raise ValueError(
                "An input length exceeds model output time steps."
            )

        if torch.any(target_lengths_cpu < 0).item():
            raise ValueError("Target lengths cannot be negative.")

        if target_lengths_cpu.sum().item() != targets.numel():
            raise ValueError(
                "Target lengths do not match concatenated targets."
            )

        if targets.numel() > 0:
            if torch.any(targets == self.blank_index).item():
                raise ValueError(
                    "CTC targets must not contain the blank token."
                )

            if (
                torch.any(targets < 0).item()
                or torch.any(targets >= num_classes).item()
            ):
                raise ValueError(
                    "Target token index is outside class range."
                )

        # Check CTC feasibility separately for each sample.
        targets_cpu = targets.detach().to(device="cpu", dtype=torch.long)
        offset = 0

        for sample_index, target_length in enumerate(
            target_lengths_cpu.tolist()
        ):
            sample_targets = targets_cpu[offset:offset + target_length]
            offset += target_length

            minimum_steps = self._minimum_ctc_timesteps(sample_targets)
            available_steps = input_lengths_cpu[sample_index].item()

            if available_steps < minimum_steps:
                raise ValueError(
                    "CTC-infeasible sample at batch index "
                    f"{sample_index}: input length={available_steps}, "
                    f"minimum required={minimum_steps}, "
                    f"target length={target_length}."
                )

        # Convert raw model logits into log probabilities.
        log_probs = F.log_softmax(logits, dim=2)

        # PyTorch CTCLoss expects [Time, Batch, Classes].
        log_probs = log_probs.permute(1, 0, 2).contiguous()

        loss = self.ctc_loss(
            log_probs,
            targets,
            input_lengths_cpu,
            target_lengths_cpu
        )

        return loss
