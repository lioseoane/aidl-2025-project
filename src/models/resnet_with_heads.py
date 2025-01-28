import torch
import torch.nn as nn
import torchvision

class resnet_with_heads(nn.Module):
    def __init__(self, num_classes, num_keypoints, use_resnet34=True):
        super(resnet_with_heads, self).__init__()

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints

        if use_resnet34:
            self.backbone = torchvision.models.resnet34(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.input_size = 512
        else:
            self.backbone = torchvision.models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.input_size = 2048

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.bbox_head = nn.Sequential(
            nn.Linear(self.input_size , 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

        self.keypoints_head = nn.Sequential(
            nn.Linear(self.input_size , 512),
            nn.ReLU(),
            nn.Linear(512, num_keypoints * 3)
        )

        self.workout_label_head = nn.Sequential(
            nn.Linear(self.input_size , 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        # resnet backbone
        x = self.backbone(x) # [batch_size, 2048 / 512, 1, 1]
        x = x.view(x.size(0), -1) # [batch_size, 2048 / 512]
        
        # bbox head, [batch_size, 4]
        bbox = self.bbox_head(x).unsqueeze(1)

        # keypoints head, [batch_size, num_keypoints, 3]
        keypoints = self.keypoints_head(x).view(-1, self.num_keypoints, 3)
        keypoints = torch.sigmoid(keypoints)  

        # visibility truncation
        visibility_thresholding = keypoints[:, :, 2]  # The visibility values are in the 3rd channel
        visibility_thresholding = (visibility_thresholding > 0.5).float()  # Set visibility to 1 or 0
        keypoints_with_visibility_thresholding = keypoints.clone()  # Clone the keypoints tensor
        keypoints_with_visibility_thresholding[:, :, 2] = visibility_thresholding # Update the visibility values

        # workout label head, [batch_size, num_classes]
        workout_label = self.workout_label_head(x) 

        return bbox, keypoints_with_visibility_thresholding, workout_label
    
    def compute_losses(self, outputs, targets):

        bbox_targets = torch.stack([target['bbox'] for target in targets]) 
        keypoints_targets = torch.stack([target['keypoints'] for target in targets]) 
        workout_label_targets = torch.stack([target['workout_label'] for target in targets]) 

        # Compute the bounding box loss
        bbox_loss = nn.L1Loss()(outputs[0], bbox_targets)

        # Compute the keypoint loss
        keypoints_loss = nn.L1Loss()(outputs[1], keypoints_targets)

        # Compute the workout label loss
        workout_label_indices = workout_label_targets.argmax(dim=1) # Get the index of the target label
        workout_label_loss = nn.CrossEntropyLoss()(outputs[2], workout_label_indices)

        return bbox_loss, keypoints_loss, workout_label_loss
    