import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Fast_RCNN(nn.Module):
    def __init__(self, num_classes, num_keypoints):
        super(Fast_RCNN, self).__init__()

        # Load the pre-trained Faster R-CNN model
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        self.backbone = self.model.backbone
        self.feature_map_size = 256  

        for param in self.backbone.parameters():
            param.requires_grad = True
        
         # Bounding Box Regression Head
        self.bbox_regression = nn.Sequential(
            nn.Conv2d(self.feature_map_size, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 4, kernel_size=1),  # Output 4 coordinates (x_min, y_min, x_max, y_max)
            nn.AdaptiveAvgPool2d(1)  # Output shape [batch_size, 4, 1, 1]
        )

        # Keypoint Predictor Head
        self.keypoint_predictor = nn.Sequential(
            nn.Conv2d(self.feature_map_size, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_keypoints * 3, kernel_size=1),  # Output (x, y, visibility) for each keypoint
            nn.AdaptiveAvgPool2d(1) # Output shape [batch_size, num_keypoints*3, 1, 1]
        )

        # Classification Head
        self.classification = nn.Sequential(
            nn.Conv2d(self.feature_map_size, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1), # Output class scores
            nn.AdaptiveAvgPool2d(1) # Output shape [batch_size, num_classes, 1, 1]
        )
        
    def forward(self, images):

        # Forward pass through the Faster R-CNN model
        feature_maps = self.backbone(images)  # Feature maps from the FPN
        feature_map_0 = feature_maps['0']  # Highest resolution (for small objects)

        # Forward pass through the custom heads
        bbox = self.bbox_regression(feature_map_0)  # Bounding boxes from feature map
        bbox = bbox.flatten(start_dim=1) # Reshape to [batch_size, 4]
        bbox = bbox.unsqueeze(1) # Reshape to [batch_size, 1, 4]

        keypoints = self.keypoint_predictor(feature_map_0)  # Keypoints from feature map
        keypoints = keypoints.flatten(start_dim=1) # Reshape to [batch_size, num_keypoints*3]
        keypoints = keypoints.view(keypoints.size(0), 17, 3) # Reshape to [batch_size, num_keypoints, 3]

        classes = self.classification(feature_map_0) # Class scores from feature map
        classes = classes.flatten(start_dim=1) # Reshape to [batch_size, num_classes]
        
        return {
            'boxes': bbox,     
            'keypoints': keypoints,  
            'labels': classes   
        }      
    
    def compute_losses(self, predicted_boxes, predicted_keypoints, predicted_labels, new_targets):

        batch_size = len(new_targets)
        target_boxes = torch.stack([new_targets[i]["boxes"] for i in range(batch_size)])  # (batch_size, 1, 4)
        target_keypoints = torch.stack([new_targets[i]["keypoints"] for i in range(batch_size)])  # (batch_size, 17, 3)
        target_labels = torch.stack([new_targets[i]["labels"] for i in range(batch_size)])  # (batch_size, 1)
    
        # 1. Compute bounding box loss (Smooth L1 loss)
        bbox_loss = F.smooth_l1_loss(predicted_boxes.squeeze(1), target_boxes.squeeze(1), reduction='mean')
        
        # 2. Compute keypoint loss (MSE loss)
        keypoint_loss = F.mse_loss(predicted_keypoints.view(-1, 3), target_keypoints.view(-1, 3), reduction='mean')

        # 3. Compute classification loss (Cross-entropy loss)
        class_loss = nn.CrossEntropyLoss()(predicted_labels.squeeze(1), target_labels.squeeze(1).long())  

        # Return the losses in a dictionary
        losses = {
            'bbox_loss': bbox_loss,
            'keypoint_loss': keypoint_loss,
            'class_loss': class_loss
        }

        return losses
