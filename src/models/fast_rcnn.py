import torch.nn as nn
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class KeypointHeadPrediction(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super(KeypointHeadPrediction, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample spatial dimensions by 2
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Final convolution to predict (x, y) for each keypoint
        self.conv_out = nn.Conv2d(64, num_keypoints * 2, kernel_size=1, stride=1)

    def forward(self, x):
        # Pass through convolutional blocks
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))

        # Final output layer
        x = self.conv_out(x)

        # Flatten spatial dimensions and reshape to [batch_size, num_keypoints, 2]
        x = x.flatten(start_dim=2)  # [batch_size, num_keypoints * 2, height * width]
        x = x.mean(dim=-1)  # Global average pooling over the spatial dimensions
        x = x.view(x.size(0), -1, 2)  # [batch_size, num_keypoints, 2]
        
        return x


def Fast_RCNN(num_classes, num_keypoints):
    # Load a pre-trained Faster R-CNN model with MobileNet backbone
    backbone = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

    # Replace Box Predictor
    in_features_box_classifier = backbone.roi_heads.box_predictor.cls_score.in_features
    backbone.roi_heads.box_predictor = FastRCNNPredictor(in_features_box_classifier, num_classes)

    # Add Custom Keypoint Predictor
    in_channels_keypoint = backbone.roi_heads.box_head.fc7.out_features
    custom_keypoint_predictor = KeypointHeadPrediction(in_channels=in_channels_keypoint, num_keypoints=num_keypoints)

    # Ensure roi_heads knows about keypoints
    backbone.roi_heads.keypoint_predictor = custom_keypoint_predictor
    backbone.roi_heads.keypoint_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=14, sampling_ratio=2
    )

    return backbone

