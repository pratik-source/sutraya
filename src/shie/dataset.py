
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.shie.config import (
    PROJECT_ROOT,
    load_config,
    validate_config,
)


DEFAULT_BACKGROUND_VALUE = 1.0


class IAMDataset(Dataset):

    def __init__(self, split, config=None):

        # ------------------------------------------
        # 1. Validate split
        # ------------------------------------------

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Invalid dataset split: {split}"
            )

        # ------------------------------------------
        # 2. Load configuration
        # ------------------------------------------

        if config is None:
            config = load_config()
        else:
            validate_config(config)

        dataset_config = config["dataset"]

        self.split = split

        # Image processing settings.
        self.image_height = dataset_config["image_height"]
        self.image_channels = dataset_config["image_channels"]
        self.horizontal_stride = dataset_config["horizontal_stride"]

        # White background: normalized pixel value.
        self.background_value = dataset_config.get(
            "background_value",
            DEFAULT_BACKGROUND_VALUE
        )

        # ------------------------------------------
        # 3. Resolve configured paths
        # ------------------------------------------

        self.dataset_root = self._resolve_path(
            dataset_config["root"]
        )

        csv_key = f"{split}_csv"

        self.csv_path = self._resolve_path(
            dataset_config[csv_key]
        )

        self.vocabulary_path = self._resolve_path(
            dataset_config["vocabulary_path"]
        )

        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"{split.capitalize()} CSV not found: "
                f"{self.csv_path}"
            )

        if not self.vocabulary_path.is_file():
            raise FileNotFoundError(
                f"Vocabulary file not found: "
                f"{self.vocabulary_path}"
            )

        # ------------------------------------------
        # 4. Load annotations
        # ------------------------------------------

        self.data = pd.read_csv(self.csv_path)

        required_columns = {
            "image_path",
            "text",
            "sample_id",
            "width",
            "height",
        }

        missing_columns = (
            required_columns - set(self.data.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing annotation columns: "
                f"{sorted(missing_columns)}"
            )

        # ------------------------------------------
        # 5. Load vocabulary
        # ------------------------------------------

        with self.vocabulary_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            self.vocabulary = json.load(file)

        self.char_to_idx = self.vocabulary["char_to_idx"]
        self.blank_idx = self.vocabulary["blank_idx"]
        self.num_classes = self.vocabulary["num_classes"]

    # ----------------------------------------------
    # Resolve paths relative to SUTRAYA project root
    # ----------------------------------------------

    @staticmethod
    def _resolve_path(path):

        path = Path(path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path.resolve()

    def __len__(self):

        return len(self.data)

    # ----------------------------------------------
    # Encode transcription into character indices
    # ----------------------------------------------

    def encode_text(self, text):

        try:

            return [
                self.char_to_idx[character]
                for character in text
            ]

        except KeyError as error:

            raise ValueError(
                f"Unknown character: {error.args[0]!r}"
            ) from error

    # ----------------------------------------------
    # Load and preprocess one IAM sample
    # ----------------------------------------------

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = self._resolve_path(
            row["image_path"]
        )

        # Read handwriting as a grayscale image.
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(image_path)

        original_height, original_width = image.shape

        if original_height == 0 or original_width == 0:
            raise ValueError(
                f"Invalid image: {image_path}"
            )

        # ------------------------------------------
        # Resize while preserving aspect ratio
        # ------------------------------------------

        resized_width = round(
            original_width
            * self.image_height
            / original_height
        )

        resized_width = max(1, resized_width)

        image = cv2.resize(
            image,
            (resized_width, self.image_height),
            interpolation=cv2.INTER_AREA
        )

        # ------------------------------------------
        # Normalize pixels
        # ------------------------------------------

        # White background = 1.0.
        # Dark handwriting = 0.0.
        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image).unsqueeze(0)

        # ------------------------------------------
        # Encode transcription
        # ------------------------------------------

        text = str(row["text"])

        target = torch.tensor(
            self.encode_text(text),
            dtype=torch.long
        )

        # ------------------------------------------
        # Calculate CTC input length
        # ------------------------------------------

        input_length = (
            resized_width // self.horizontal_stride
        )

        # CTC requires an additional timestep
        # between consecutive repeated characters.
        repeated_count = sum(
            text[index] == text[index - 1]
            for index in range(1, len(text))
        )

        minimum_length = len(text) + repeated_count

        if input_length < minimum_length:

            raise ValueError(
                f"CTC-infeasible sample: {row['sample_id']} "
                f"(input={input_length}, "
                f"required={minimum_length})"
            )

        # ------------------------------------------
        # Return sample
        # ------------------------------------------

        return {
            "image": image,
            "target": target,
            "input_length": input_length,
            "target_length": len(target),
            "text": text,
            "sample_id": row["sample_id"],
        }


# ==================================================
# BATCH COLLATION
# ==================================================

def collate_iam_batch(
    batch,
    image_height=32,
    horizontal_stride=2,
    background_value=DEFAULT_BACKGROUND_VALUE,
    image_channels=1,
):

    if not batch:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    if image_channels != 1:
        raise ValueError(
            "SHIE currently supports grayscale images only."
        )

    batch_size = len(batch)

    # ----------------------------------------------
    # Find maximum image width
    # ----------------------------------------------

    max_width = max(
        item["image"].shape[-1]
        for item in batch
    )

    # Ensure padded width is divisible by CNN stride.
    remainder = max_width % horizontal_stride

    if remainder:
        max_width += horizontal_stride - remainder

    # ----------------------------------------------
    # Create white padded image tensor
    # ----------------------------------------------

    images = torch.full(
        (
            batch_size,
            image_channels,
            image_height,
            max_width,
        ),
        fill_value=background_value,
        dtype=torch.float32
    )

    targets = []
    input_lengths = []
    target_lengths = []
    texts = []
    sample_ids = []

    # ----------------------------------------------
    # Copy images and collect targets
    # ----------------------------------------------

    for index, item in enumerate(batch):

        image = item["image"]

        if image.shape[0] != image_channels:
            raise ValueError(
                f"Unexpected image channels: {image.shape[0]}"
            )

        if image.shape[1] != image_height:
            raise ValueError(
                f"Unexpected image height: {image.shape[1]}"
            )

        width = image.shape[-1]

        images[index, :, :, :width] = image

        targets.append(item["target"])
        input_lengths.append(item["input_length"])
        target_lengths.append(item["target_length"])
        texts.append(item["text"])
        sample_ids.append(item["sample_id"])

    targets = torch.cat(targets)

    return {
        "images": images,
        "targets": targets,
        "input_lengths": torch.tensor(
            input_lengths,
            dtype=torch.long
        ),
        "target_lengths": torch.tensor(
            target_lengths,
            dtype=torch.long
        ),
        "texts": texts,
        "sample_ids": sample_ids,
    }
