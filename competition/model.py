import torch
import torch.nn as nn


class GateCornerCNN(nn.Module):

    def __init__(self, max_gates=10, num_outputs_gate=12, input_channels=3):

        super().__init__()

        self.num_gates = max_gates
        self.num_outputs_per_gate = num_outputs_gate

        # Encoder definition
        self.encoder = nn.Sequential(
            # Convolutional layer 1
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Conv Layer 2
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Downsample by 2
            # Conv Layer 3
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Downsample by 2
            # Conv Layer 4
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Downsample by 2
        )

        # Flatten the convolutional output to feed into the fully connected layers
        self.flatten = nn.Flatten()

        self.fc_layers = nn.Sequential(
            nn.LazyLinear(
                512
            ),  # Adjust input size based on the image resolution and convolutional layers
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                256, max_gates * num_outputs_gate
            ),  # Final layer to output 10 gates * 12 values
        )

    def forward(self, x):
        # Forward pass through the convolutional layers (encoder)
        x = self.encoder(x)

        # Flatten the output for the fully connected layers
        x = self.flatten(x)

        # Forward pass through the fully connected layers
        x = self.fc_layers(x)

        # Reshape to [batch_size, 10 gates, 12 outputs per gate]
        x = x.view(-1, self.num_gates, self.num_outputs_per_gate)

        return x
