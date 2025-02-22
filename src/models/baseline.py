import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights
import timm  # For HRNet

class baseline(nn.Module):
    def __init__(self, num_classes, num_keypoints, backbone='resnet50'):
        super(baseline, self).__init__()

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.backbone_label = backbone

        if backbone == 'resnet50':

            self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.input_size = 2048

            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        if backbone == 'hrnet_w32':
            self.backbone = timm.create_model('hrnet_w32', pretrained=True, features_only=True)
            self.input_size = 1024  


        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),  
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, 4),
            nn.Sigmoid()
        )

        self.keypoints_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),  
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, num_keypoints * 3),
        )

        self.workout_label_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),  
            nn.Linear(self.input_size , self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, num_classes)
        )


    def forward(self, x):

        if self.backbone_label == 'resnet50':
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            feat1 = self.backbone.layer1(x)  # Feature Map after Layer 1
            feat2 = self.backbone.layer2(feat1)  # Feature Map after Layer 2
            feat3 = self.backbone.layer3(feat2)  # Feature Map after Layer 3
            feat = self.backbone.layer4(feat3)  # Feature Map after Layer 4 (Final)

        elif  self.backbone_label == 'hrnet_w32':
            features = self.backbone(x)  # HRNet outputs multiple scales; we use the last one
            feat = features[-1]  # Highest resolution feature

        # bbox head, [batch_size, 4]
        bbox = self.bbox_head(feat)

        # keypoints heatmaps, [batch_size, num_keypoints, 3]
        keypoints = self.keypoints_head(feat)
        keypoints = self.keypoints_head(feat).view(-1, self.num_keypoints, 3)
        keypoints = torch.sigmoid(keypoints)  

        # workout label head, [batch_size, num_classes]
        workout_label = self.workout_label_head(feat)

        return bbox, keypoints, workout_label
    
    def compute_losses(self, outputs, targets):
        # Extract targets
        bbox_targets = torch.stack([target['boxes'] for target in targets]) 
        keypoints_targets = torch.stack([target['keypoints'] for target in targets]) 
        workout_label_targets = torch.stack([target['workout_labels'] for target in targets]) 
            
        # Check if the tensor is batch_size, 1, bbox/keypoints or batch_sizem bbox/keypoints
        if bbox_targets.shape[1] == 1:
            bbox_targets = bbox_targets.squeeze(1)

        if keypoints_targets.shape[1] == 1:
            keypoints_targets = keypoints_targets.squeeze(1)
            
        # Confidence scores boolean
        use_confience_scores = keypoints_targets.shape[-1] == 3

        # If confidence scores, use it
        if use_confience_scores:
            confidence_scores  = keypoints_targets[:, :, 2]
        else:
            confidence_scores = torch.ones_like(keypoints_targets[:, :, 0])  # If no confidence socres, assume all keypoints are visible

        # Convert confidence scores to visibility mask (0 = ignore, 1 = include)
        visibility_mask = (confidence_scores > 0.5).float()  # If confidence > 0.3, include in loss

        # Compute the bounding box loss
        bbox_loss = nn.MSELoss()(outputs[0], bbox_targets)

        # Compute the keypoint loss
        keypoints_loss = nn.MSELoss(reduction='none')(outputs[1][:, :, :2], keypoints_targets[:, :, :2])
            
        # Apply visibility mask if available (ignores keypoints where visibility = 0)
        keypoints_loss = keypoints_loss * visibility_mask.unsqueeze(-1)  # Keep shape (B, num_keypoints, 2)
        keypoints_loss = keypoints_loss.sum() / (visibility_mask.sum() + 1e-6) # Avoid division by 0

        # Compute the workout label loss
        workout_label_indices = workout_label_targets.argmax(dim=1)  # Get the index of the target label
        workout_label_loss = nn.CrossEntropyLoss()(outputs[2], workout_label_indices)

        return bbox_loss, keypoints_loss, workout_label_loss