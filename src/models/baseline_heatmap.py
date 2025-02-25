import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights
import timm  # For HRNet


class baseline_heatmap(nn.Module):
    def __init__(self, num_classes, num_keypoints, backbone='resnet50'):
        super(baseline_heatmap, self).__init__()

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.backbone_label = backbone

        if backbone == 'resnet50':
            self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.input_size = 2048

        if backbone == 'hrnet_w32':
            self.backbone = timm.create_model('hrnet_w32', pretrained=True, features_only=True)
            self.input_size = 1024 

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),  
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, 4),
            nn.Sigmoid()
        )

        self.keypoints_head = nn.Sequential(
            nn.ConvTranspose2d(self.input_size, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, num_keypoints, kernel_size=1),
        )

        self.workout_label_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),  
            nn.Linear(self.input_size , self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, num_classes)
        )

        # Initialize only the heads, not the backbone:
        self.keypoints_head.apply(baseline_heatmap.init_weights)

    def forward(self, x):

        if self.backbone_label == 'resnet50':
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            feat1 = self.backbone.layer1(x)  # low-level features
            feat2 = self.backbone.layer2(feat1)  # layer2 features
            feat3 = self.backbone.layer3(feat2)  # layer3 features
            feat4 = self.backbone.layer4(feat3)  # layer4 features

            # 2048x7x7

        elif self.backbone_label == 'hrnet_w32':
            features = self.backbone(x)  # HRNet outputs multiple scales; we use the last one
            feat = features[-1]  # Highest resolution feature
        
        # bbox head, [batch_size, 4]
        bbox = self.bbox_head(feat4)

        # workout label head, [batch_size, num_classes]
        workout_label = self.workout_label_head(feat4)
    
        # keypoints heatmaps, [batch_size, num_keypoints, 3]
        keypoints = self.keypoints_head(feat4)

        return bbox, keypoints, workout_label
        
    def compute_losses(self, outputs, targets):
        # Extract targets
        bbox_targets = torch.stack([target['boxes'] for target in targets]) 
        keypoints_targets = torch.stack([target['keypoints'] for target in targets]) 
        workout_label_targets = torch.stack([target['workout_labels'] for target in targets]) 

        # Compute the bounding box loss
        bbox_loss = nn.MSELoss()(outputs[0], bbox_targets)

        # Compute the keypoint loss
        keypoints_loss = nn.MSELoss()(outputs[1], keypoints_targets)

        # Compute the workout label loss
        workout_label_indices = workout_label_targets.argmax(dim=1)  # Get the index of the target label
        workout_label_loss = nn.CrossEntropyLoss()(outputs[2], workout_label_indices)

        return bbox_loss, keypoints_loss, workout_label_loss
    
    @staticmethod
    def init_weights(m):
        # For convolutional layers
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # mean defaults to 0, but you can be explicit:
            nn.init.normal_(m.weight, mean=0, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # For BatchNorm layers
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)