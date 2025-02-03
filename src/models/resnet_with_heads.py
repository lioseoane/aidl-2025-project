import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet34_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import ViT_B_16_Weights

class resnet_with_heads(nn.Module):
    def __init__(self, num_classes, num_keypoints, backbone='resnet34'):
        super(resnet_with_heads, self).__init__()

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.backbone_label = backbone

        if self.backbone_label == 'resnet34':
            self.backbone = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.input_size = 512
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone[-1].parameters():
                param.requires_grad = True

        elif self.backbone_label == 'resnet50':
            self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.input_size = 2048
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone[-1].parameters():
                param.requires_grad = True

        elif self.backbone_label == 'vit_b_16':
            self.backbone = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.backbone.heads = nn.Identity()
            self.input_size = 768

        # Shared Intermediate Layer
        #self.shared_fc = nn.Sequential(
            #nn.Linear(self.input_size, self.input_size),
            #nn.ReLU(),
            #nn.Dropout(0.3)  
        #)

        self.bbox_head = nn.Sequential(
            nn.Linear(self.input_size , self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, 4)
        )

        self.keypoints_head = nn.Sequential(
            nn.Linear(self.input_size , self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, num_keypoints * 3)
        )

        self.workout_label_head = nn.Sequential(
            nn.Linear(self.input_size , self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, num_classes)
        )

        self.log_vars = torch.nn.Parameter(torch.zeros(3))  # One per task

    def forward(self, x):

        # resnet backbone
        x = self.backbone(x) # [batch_size, 2048 / 512, 1, 1]

        if self.backbone_label == 'vit_b_16':
            x = nn.ReLU()(x)
        else:
            x = x.view(x.size(0), -1) # [batch_size, 2048 / 512]

        #x = self.shared_fc(x)
        
        # bbox head, [batch_size, 4]
        bbox = self.bbox_head(x)

        # keypoints head, [batch_size, num_keypoints, 3]
        keypoints = self.keypoints_head(x).view(-1, self.num_keypoints, 3)
        keypoints = torch.sigmoid(keypoints)  
#
        # workout label head, [batch_size, num_classes]
        workout_label = self.workout_label_head(x)

        return bbox, keypoints, workout_label
    
    def compute_losses(self, outputs, targets, weighted=False):

        bbox_targets = torch.stack([target['bbox'] for target in targets]) 
        keypoints_targets = torch.stack([target['keypoints'] for target in targets]) 
        workout_label_targets = torch.stack([target['workout_label'] for target in targets]) 

        visibility_mask = keypoints_targets[:, :, 2]

        # Compute the bounding box loss
        bbox_loss = nn.L1Loss()(outputs[0], bbox_targets)

        # Compute the keypoint loss
        keypoints_loss = nn.L1Loss(reduction='none')(outputs[1][:, :, :2], keypoints_targets[:, :, :2])
        # Apply visibility mask (ignores keypoints where visibility = 0)
        keypoints_loss = keypoints_loss * visibility_mask.unsqueeze(-1)  # Keep shape (B, num_keypoints, 2)
        keypoints_loss = keypoints_loss.sum() / visibility_mask.sum().clamp(min=1)

        # Compute the workout label loss
        workout_label_indices = workout_label_targets.argmax(dim=1) # Get the index of the target label
        workout_label_loss = nn.CrossEntropyLoss()(outputs[2], workout_label_indices)

        if weighted == True:
            # Weighting losses dynamically
            task_losses = torch.stack([bbox_loss, keypoints_loss, workout_label_loss])
            loss_weights = torch.exp(-self.log_vars)
            weighted_loss = (loss_weights * task_losses).sum() + self.log_vars.sum()  # Regularization

            return weighted_loss
        else:
            return bbox_loss, keypoints_loss, workout_label_loss
    