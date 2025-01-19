import torch
import torch.nn as nn

class KeypointModel(nn.Module):
    def __init__(self, num_keypoints=16, input_size=128):
        super(KeypointModel, self).__init__()


        # Image size
        self.input_size = input_size

        # Define CNNs block
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        # Dynamically calculate the size of the feature map after convolution
        feature_map_size = input_size // (2 ** 3)  # 3 maxpool layers, each reducing size by 2

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feature_map_size * feature_map_size, 512), nn.ReLU(),
            nn.Linear(512, num_keypoints * 2)  # Predict x, y coordinates for each keypoint
        )

    def forward(self, x):
        # Pass the input (images) through the convolutional layers
        x = self.conv_layers(x)
        # Pass through the fully connected layers
        return self.fc_layers(x)
