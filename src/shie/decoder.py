
import json
from pathlib import Path

import torch


# Project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SHIE vocabulary file.
DEFAULT_VOCABULARY_PATH = (
    PROJECT_ROOT / "configs" / "shie_vocabulary.json"
)


class CTCDecoder:
    """
    Greedy CTC decoder for SHIE.

    Converts model predictions into readable text by:
    1. Removing consecutive repeated tokens.
    2. Removing CTC blank tokens.
    """

    def __init__(self, vocabulary_path=None):

        if vocabulary_path is None:
            vocabulary_path = DEFAULT_VOCABULARY_PATH

        vocabulary_path = Path(vocabulary_path)

        if not vocabulary_path.is_absolute():
            vocabulary_path = PROJECT_ROOT / vocabulary_path

        if not vocabulary_path.is_file():
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocabulary_path}"
            )

        # Load vocabulary.
        with open(vocabulary_path, "r", encoding="utf-8") as file:
            vocabulary = json.load(file)

        self.blank_idx = vocabulary["blank_idx"]

        # Convert JSON string keys into integer keys.
        self.idx_to_char = {
            int(index): character
            for index, character in vocabulary["idx_to_char"].items()
        }

        self.char_to_idx = vocabulary["char_to_idx"]

        self.num_classes = vocabulary["num_classes"]

        # Validate vocabulary.
        if self.blank_idx != 0:
            raise ValueError(
                "SHIE currently expects CTC blank index 0."
            )

        if self.blank_idx in self.idx_to_char:
            raise ValueError(
                "CTC blank index must not represent a character."
            )

        if len(self.idx_to_char) + 1 != self.num_classes:
            raise ValueError(
                "Vocabulary character count does not match num_classes."
            )

        if set(self.idx_to_char) != (
            set(range(self.num_classes)) - {self.blank_idx}
        ):
            raise ValueError(
                "Vocabulary indices must be contiguous."
            )

        print("SHIE CTC Decoder initialized successfully.")

    def decode_sequence(self, token_ids):
        """
        Decode one sequence of predicted token IDs.

        Args:
            token_ids: List or iterable of predicted token IDs.

        Returns:
            str: Decoded text.
        """

        decoded_characters = []

        previous_token = None

        for token in token_ids:

            token = int(token)

            if token < 0 or token >= self.num_classes:
                raise ValueError(
                    f"Invalid token index: {token}"
                )

            # CTC rule 1:
            # Remove consecutive repeated tokens.
            if token != previous_token:

                # CTC rule 2:
                # Ignore blank tokens.
                if token != self.blank_idx:

                    character = self.idx_to_char.get(token)

                    if character is None:
                        raise ValueError(
                            f"Unknown vocabulary index: {token}"
                        )

                    decoded_characters.append(character)

            # Update previous token, including blank.
            previous_token = token

        return "".join(decoded_characters)

    def decode_batch(self, logits, input_lengths=None):
        """
        Decode a batch of model predictions.

        Args:
            logits: Tensor with shape [Batch, Time, Classes].
            input_lengths: Valid time steps for each sample.

        Returns:
            list[str]: Decoded text for each sample.
        """

        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a PyTorch tensor.")

        if logits.ndim != 3:
            raise ValueError(
                "Expected logits shape [Batch, Time, Classes]."
            )

        batch_size, time_steps, num_classes = logits.shape

        if num_classes != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, "
                f"received {num_classes}."
            )

        if not torch.isfinite(logits).all():
            raise ValueError("Logits contain non-finite values.")

        # Select the highest-scoring class at each time step.
        predictions = logits.detach().argmax(dim=2)

        predictions = predictions.cpu()

        # Use full time dimension if lengths are not provided.
        if input_lengths is None:
            input_lengths = [time_steps] * batch_size

        elif isinstance(input_lengths, torch.Tensor):
            input_lengths = input_lengths.detach().cpu().tolist()

        if len(input_lengths) != batch_size:
            raise ValueError(
                "Number of input lengths must match batch size."
            )

        decoded_results = []

        for index in range(batch_size):

            valid_length = int(input_lengths[index])

            if valid_length < 0 or valid_length > time_steps:
                raise ValueError(
                    f"Invalid input length: {valid_length}"
                )

            sequence = predictions[index, :valid_length]

            decoded_text = self.decode_sequence(sequence)

            decoded_results.append(decoded_text)

        return decoded_results
