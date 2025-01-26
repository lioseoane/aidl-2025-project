import torch
import torchvision

# Load pre-trained Keypoint R-CNN model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)

# Create a dummy input image (batch_size, channels, height, width)
dummy_input = torch.rand(1, 3, 224, 224)  # Assuming input size of 224x224

# Pass the input through the backbone (ResNet + FPN)
features = model.backbone(dummy_input)

# Print the feature map size
for feature in features.values():
    print("Feature map size:", feature.shape)