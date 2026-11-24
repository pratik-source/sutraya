
import torch
import torch.nn as nn

from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


class SHIEModel(nn.Module):

    def __init__(self, num_classes: int, config=None):

        super().__init__()

        # ==================================================
        # 1. DEFAULT ARCHITECTURE
        # ==================================================

        # These defaults preserve the previously tested SHIE
        # architecture when config is not supplied.

        image_height = 32
        image_channels = 1
        horizontal_stride = 2

        cnn_channels = [32, 64, 128, 256]

        hidden_size = 256
        num_layers = 2
        bidirectional = True

        # ==================================================
        # 2. READ CONFIGURATION
        # ==================================================

        if config is not None:

            if not isinstance(config, dict):
                raise TypeError(
                    "config must be a configuration dictionary."
                )

            try:
                dataset_config = config["dataset"]
                model_config = config["model"]

                image_height = dataset_config["image_height"]
                image_channels = dataset_config["image_channels"]
                horizontal_stride = dataset_config["horizontal_stride"]

                cnn_channels = model_config["cnn"]["channels"]

                hidden_size = model_config["lstm"]["hidden_size"]
                num_layers = model_config["lstm"]["num_layers"]
                bidirectional = model_config["lstm"]["bidirectional"]

            except (KeyError, TypeError) as error:

                raise ValueError(
                    "Configuration is missing required SHIE model settings."
                ) from error

            configured_classes = model_config["num_classes"]

            if configured_classes != num_classes:
                raise ValueError(
                    "num_classes does not match model configuration."
                )

            classifier_config = model_config["classifier"]

            if classifier_config["output_size"] != num_classes:
                raise ValueError(
                    "Classifier output_size must match num_classes."
                )

        # ==================================================
        # 3. VALIDATE MODEL SETTINGS
        # ==================================================

        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes <= 1
        ):
            raise ValueError(
                "num_classes must be an integer greater than 1."
            )

        if image_height != 32:
            raise ValueError(
                "Current SHIE CNN architecture supports image_height=32 only."
            )

        if image_channels != 1:
            raise ValueError(
                "Current SHIE architecture supports grayscale images only."
            )

        if horizontal_stride != 2:
            raise ValueError(
                "Current SHIE CNN architecture supports horizontal_stride=2 only."
            )

        if (
            not isinstance(cnn_channels, (list, tuple))
            or len(cnn_channels) != 4
        ):
            raise ValueError(
                "CNN channels must contain exactly four values."
            )

        for channel_count in cnn_channels:

            if (
                not isinstance(channel_count, int)
                or isinstance(channel_count, bool)
                or channel_count <= 0
            ):
                raise ValueError(
                    "Every CNN channel count must be a positive integer."
                )

        if (
            not isinstance(hidden_size, int)
            or isinstance(hidden_size, bool)
            or hidden_size <= 0
        ):
            raise ValueError(
                "LSTM hidden_size must be a positive integer."
            )

        if (
            not isinstance(num_layers, int)
            or isinstance(num_layers, bool)
            or num_layers <= 0
        ):
            raise ValueError(
                "LSTM num_layers must be a positive integer."
            )

        if not isinstance(bidirectional, bool):
            raise ValueError(
                "LSTM bidirectional must be a boolean."
            )

        # ==================================================
        # 4. STORE ARCHITECTURE SETTINGS
        # ==================================================

        self.num_classes = num_classes

        self.input_height = image_height
        self.input_channels = image_channels

        self.horizontal_stride = horizontal_stride

        self.cnn_channels = list(cnn_channels)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # ==================================================
        # 5. CNN FEATURE EXTRACTOR
        # ==================================================

        c1, c2, c3, c4 = self.cnn_channels

        self.cnn = nn.Sequential(

            # ----------------------------------------------
            # Block 1: Reduce height and width by half
            # ----------------------------------------------

            nn.Conv2d(
                self.input_channels,
                c1,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(c1),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # ----------------------------------------------
            # Block 2: Reduce height only
            # ----------------------------------------------

            nn.Conv2d(
                c1,
                c2,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(c2),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=(2, 1)
            ),

            # ----------------------------------------------
            # Block 3: Reduce height only
            # ----------------------------------------------

            nn.Conv2d(
                c2,
                c3,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(c3),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=(2, 1)
            ),

            # ----------------------------------------------
            # Block 4: Reduce height only
            # ----------------------------------------------

            nn.Conv2d(
                c3,
                c4,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(c4),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=(2, 1)
            ),
        )

        # ==================================================
        # 6. BIDIRECTIONAL LSTM
        # ==================================================

        self.lstm = nn.LSTM(

            input_size=c4,

            hidden_size=self.hidden_size,

            num_layers=self.num_layers,

            batch_first=True,

            bidirectional=self.bidirectional,
        )

        # ==================================================
        # 7. CLASSIFIER
        # ==================================================

        direction_multiplier = (
            2 if self.bidirectional else 1
        )

        classifier_input_size = (
            self.hidden_size * direction_multiplier
        )

        self.classifier = nn.Linear(
            classifier_input_size,
            self.num_classes
        )

    # ======================================================
    # 8. FORWARD PROPAGATION
    # ======================================================

    def forward(self, x, input_lengths=None):

        # ----------------------------------------------
        # Validate input tensor
        # ----------------------------------------------

        if x.ndim != 4:
            raise ValueError(
                "Expected input shape [B, 1, 32, W]."
            )

        if (
            x.shape[1] != self.input_channels
            or x.shape[2] != self.input_height
        ):
            raise ValueError(
                "Input must contain grayscale images "
                "with height 32."
            )

        if x.shape[-1] < self.horizontal_stride:
            raise ValueError(
                "Input image width must be at least 2."
            )

        # ----------------------------------------------
        # Step 1: CNN feature extraction
        # ----------------------------------------------

        x = self.cnn(x)

        # Expected:
        # [B, 256, 2, W/2] with current configuration.

        # ----------------------------------------------
        # Step 2: Convert CNN output into sequence format
        # ----------------------------------------------

        x = x.permute(0, 3, 1, 2)

        # Shape:
        # [B, W/2, 2, 256]

        # ----------------------------------------------
        # Step 3: Reduce remaining feature height
        # ----------------------------------------------

        x = x.mean(dim=3)

        # Shape:
        # [B, W/2, 256]

        # ----------------------------------------------
        # Step 4: Process sequence using BiLSTM
        # ----------------------------------------------

        if input_lengths is not None:

            if isinstance(input_lengths, torch.Tensor):

                lengths = input_lengths.detach().to(
                    device="cpu",
                    dtype=torch.long
                )

            else:

                lengths = torch.tensor(
                    input_lengths,
                    dtype=torch.long
                )

            if lengths.ndim != 1:
                raise ValueError(
                    "input_lengths must be a 1D sequence."
                )

            if lengths.numel() != x.shape[0]:
                raise ValueError(
                    "Number of input lengths must match batch size."
                )

            if torch.any(lengths <= 0):
                raise ValueError(
                    "All input lengths must be positive."
                )

            if torch.any(lengths > x.shape[1]):
                raise ValueError(
                    "Input length exceeds CNN output sequence length."
                )

            packed_x = pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )

            packed_output, _ = self.lstm(packed_x)

            x, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=x.shape[1]
            )

        else:

            x, _ = self.lstm(x)

        # ----------------------------------------------
        # Step 5: Character classification
        # ----------------------------------------------

        logits = self.classifier(x)

        # Shape:
        # [B, T, num_classes]

        return logits

    # ======================================================
    # 9. CALCULATE CNN OUTPUT SEQUENCE LENGTHS
    # ======================================================

    def get_output_lengths(self, input_widths):

        return input_widths // self.horizontal_stride
