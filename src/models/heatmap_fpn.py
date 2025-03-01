import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

class heatmap_fpn(nn.Module):
    def __init__(self, num_classes, num_keypoints, backbone='resnet50'):
        super(heatmap_fpn, self).__init__()

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.backbone_label = backbone
        self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.input_size = 2048

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Feature Pyramid Lateral Connections
        self.lateral1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.lateral2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.lateral3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.lateral4 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Up-sampling layers to combine feature maps
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Final smoothing convolutions
        self.smooth1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),  
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, 4),
            nn.Sigmoid()
        )

        self.keypoints_head = nn.Sequential(
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
        self.keypoints_head.apply(heatmap_fpn.init_weights)

    def forward(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        feat1 = self.backbone.layer1(x)  # low-level features
        feat2 = self.backbone.layer2(feat1)  # layer2 features
        feat3 = self.backbone.layer3(feat2)  # layer3 features
        feat4 = self.backbone.layer4(feat3)  # layer4 features

        p4 = self.lateral4(feat4)  # (B, 256, H/32, W/32)
        p3 = self.lateral3(feat3) + self.upsample(p4)  # (B, 256, H/16, W/16)
        p3 = self.smooth3(p3)

        p2 = self.lateral2(feat2) + self.upsample(p3)  # (B, 256, H/8, W/8)
        p2 = self.smooth2(p2)

        p1 = self.lateral1(feat1) + self.upsample(p2)  # (B, 256, H/4, W/4)
        p1 = self.smooth1(p1)

        # Final feature map to use for heads
        final_feat = p1  # Highest resolution feature
        
        # bbox head, [batch_size, 4]
        bbox = self.bbox_head(feat4)
        # workout label head, [batch_size, num_classes]
        workout_label = self.workout_label_head(feat4)
        # keypoints heatmaps, [batch_size, num_keypoints, 3]
        keypoints = self.keypoints_head(final_feat)

        return bbox, keypoints, workout_label
        
    def compute_losses(self, outputs, targets):
        # Extract targets
        bbox_targets = torch.stack([target['boxes'] for target in targets]) 
        keypoints_targets = torch.stack([target['keypoints'] for target in targets]) 
        workout_label_targets = torch.stack([target['workout_labels'] for target in targets]) 
        confidences_targets = torch.stack([target['confidences'] for target in targets]) 

        # Compute the bounding box loss
        bbox_loss = nn.MSELoss()(outputs[0], bbox_targets)

        # Compute the keypoint loss
        keypoints_loss = nn.MSELoss(reduction='none')(outputs[1], keypoints_targets)
        confidence_mask = confidences_targets[:, :, None]
        keypoints_loss = keypoints_loss * confidence_mask
        keypoints_loss = keypoints_loss.sum() / (confidence_mask.sum() + 1e-6) 

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