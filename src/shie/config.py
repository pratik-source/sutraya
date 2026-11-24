
import json
import math
from pathlib import Path

import yaml


# Project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default SHIE configuration file.
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "shie" / "shie_config.yaml"
)


def resolve_project_path(path):
    """
    Resolve a configuration path relative to the SUTRAYA project root.

    Absolute paths are returned unchanged.
    Relative paths are resolved from PROJECT_ROOT.
    """
    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_config(config_path=None):
    """
    Load and validate the SHIE configuration.

    Args:
        config_path: Optional custom YAML configuration path.

    Returns:
        dict: Validated configuration dictionary.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = resolve_project_path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    # Read YAML configuration.
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML dictionary."
        )

    # Validate configuration.
    validate_config(config)

    return config


def _require_positive_integer(value, name):
    """Validate a positive integer, excluding boolean values."""
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )


def _require_existing_file(path_value, name):
    """Resolve a configured path and verify that it is a file."""
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(
            f"{name} must be a non-empty path."
        )

    path = resolve_project_path(path_value)

    if not path.is_file():
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    return path


def validate_config(config):
    """
    Validate all required SHIE configuration settings.

    Current architectural constraints:
        - Image height: 32
        - Image channels: 1 (grayscale)
        - Horizontal stride: 2
    """

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must be a dictionary."
        )

    # ------------------------------------------
    # 1. Validate required configuration sections
    # ------------------------------------------

    required_sections = [
        "project",
        "dataset",
        "model",
        "training",
        "hardware",
        "checkpoint",
        "logging",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

        if not isinstance(config[section], dict):
            raise ValueError(
                f"Configuration section must be a dictionary: {section}"
            )

    dataset = config["dataset"]
    model = config["model"]
    training = config["training"]
    hardware = config["hardware"]

    # ------------------------------------------
    # 2. Validate dataset configuration
    # ------------------------------------------

    required_dataset_fields = [
        "name",
        "root",
        "train_csv",
        "validation_csv",
        "test_csv",
        "image_height",
        "image_channels",
        "horizontal_stride",
        "vocabulary_path",
    ]

    for field in required_dataset_fields:
        if field not in dataset:
            raise ValueError(
                f"Missing dataset configuration: {field}"
            )

    if dataset["image_height"] != 32:
        raise ValueError(
            "SHIE currently requires image_height = 32."
        )

    if dataset["image_channels"] != 1:
        raise ValueError(
            "SHIE currently requires image_channels = 1 "
            "(grayscale images)."
        )

    if dataset["horizontal_stride"] != 2:
        raise ValueError(
            "SHIE currently requires horizontal_stride = 2."
        )

    # Validate dataset root.
    dataset_root = resolve_project_path(dataset["root"])

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root directory not found: {dataset_root}"
        )

    # Validate all dataset CSV files.
    _require_existing_file(
        dataset["train_csv"],
        "Training CSV",
    )

    _require_existing_file(
        dataset["validation_csv"],
        "Validation CSV",
    )

    _require_existing_file(
        dataset["test_csv"],
        "Test CSV",
    )

    # ------------------------------------------
    # 3. Validate model configuration
    # ------------------------------------------

    num_classes = model.get("num_classes")

    if (
        not isinstance(num_classes, int)
        or isinstance(num_classes, bool)
        or num_classes <= 1
    ):
        raise ValueError(
            "Model num_classes must be an integer greater than 1."
        )

    classifier = model.get("classifier")

    if not isinstance(classifier, dict):
        raise ValueError(
            "Model classifier configuration must be a dictionary."
        )

    classifier_size = classifier.get("output_size")

    if classifier_size != num_classes:
        raise ValueError(
            "Classifier output_size must match model num_classes."
        )

    # ------------------------------------------
    # 4. Validate training configuration
    # ------------------------------------------

    batch_size = training.get("batch_size")
    epochs = training.get("epochs")
    learning_rate = training.get("learning_rate")

    _require_positive_integer(batch_size, "batch_size")
    _require_positive_integer(epochs, "epochs")

    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError(
            "learning_rate must be a finite positive number."
        )

    # ------------------------------------------
    # 5. Validate hardware configuration
    # ------------------------------------------

    if hardware.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "Hardware device must be auto, cpu or cuda."
        )

    # ------------------------------------------
    # 6. Validate vocabulary configuration
    # ------------------------------------------

    vocabulary_path = _require_existing_file(
        dataset["vocabulary_path"],
        "Vocabulary file",
    )

    try:
        with vocabulary_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            vocabulary = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid vocabulary JSON: {vocabulary_path}"
        ) from error

    if not isinstance(vocabulary, dict):
        raise ValueError(
            "Vocabulary file must contain a JSON object."
        )

    vocabulary_classes = vocabulary.get("num_classes")
    blank_index = vocabulary.get("blank_idx")
    char_to_idx = vocabulary.get("char_to_idx")

    if vocabulary_classes != num_classes:
        raise ValueError(
            f"Vocabulary has {vocabulary_classes} classes, "
            f"but model expects {num_classes}."
        )

    if training.get("blank_index") != blank_index:
        raise ValueError(
            "Training blank_index does not match vocabulary."
        )

    if not isinstance(char_to_idx, dict):
        raise ValueError(
            "Vocabulary char_to_idx must be a dictionary."
        )

    if len(char_to_idx) + 1 != vocabulary_classes:
        raise ValueError(
            "Vocabulary character count does not match num_classes."
        )

    # The current SHIE vocabulary reserves index 0 for CTC blank.
    if blank_index != 0:
        raise ValueError(
            "SHIE currently requires the CTC blank index to be 0."
        )

    character_indices = list(char_to_idx.values())

    if (
        any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in character_indices
        )
        or sorted(character_indices) != list(
            range(1, vocabulary_classes)
        )
    ):
        raise ValueError(
            "Vocabulary character indices must be unique and "
            "contiguous, starting from 1."
        )

    # ------------------------------------------
    # 7. Validation completed
    # ------------------------------------------

    print("SHIE configuration validation successful.")

    return True
