import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.image_list import ImageList


class resnet_with_heads(nn.Module):
    def __init__(self, num_classes, num_keypoints, model='resnet50'):
        super(resnet_with_heads, self).__init__()

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.model_label = model

        if self.model_label == 'resnet50':
            self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.input_size = 2048
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone[6].parameters():
                param.requires_grad = True

            self.bbox_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Flatten(),  
                nn.Linear(self.input_size , self.input_size),
                nn.ReLU(),
                nn.Linear(self.input_size, 4)
            )

            self.keypoints_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Flatten(),  
                nn.Linear(self.input_size , self.input_size),
                nn.ReLU(),
                nn.Linear(self.input_size, num_keypoints * 3)
            )

            self.workout_label_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Flatten(),  
                nn.Linear(self.input_size , self.input_size),
                nn.ReLU(),
                nn.Linear(self.input_size, num_classes)
            )

        elif self.model_label == 'keypoint-rcnn':
            self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
                weights= KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
                weights_backbone = ResNet50_Weights.IMAGENET1K_V1,
                num_keypoints= self.num_keypoints,
                num_classes = 2,
                trainable_backbone_layers = 0)
            
            self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(self.in_features, 2)

            in_features_keypoint = self.model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
            self.model.roi_heads.keypoint_predictor = torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor(in_features_keypoint, num_keypoints)

            self.workout_label_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Flatten(),  
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )

    def forward(self, x, targets=None):

        if self.model_label == 'resnet50':

            # resnet backbone
            x = self.backbone(x) 

            # bbox head, [batch_size, 4]
            bbox = self.bbox_head(x)

            # keypoints head, [batch_size, num_keypoints, 3]
            keypoints = self.keypoints_head(x).view(-1, self.num_keypoints, 3)
            keypoints = torch.sigmoid(keypoints)  

            # workout label head, [batch_size, num_classes]
            workout_label = self.workout_label_head(x)

            return bbox, keypoints, workout_label
        
        
        elif self.model_label == 'keypoint-rcnn':

            if self.training:  # Training mode

                image_list = ImageList(x, [(image.shape[1], image.shape[2]) for image in x])
                image_sizes = image_list.image_sizes 

                # Denormalize boxes and keypoints for each image in the batch
                denormalized_bbox = []
                denormalized_keypoints = []
                labels = []
                workout_labels = []

                for i, target in enumerate(targets): 
                    norm_boxes = target["boxes"]  # Normalized boxes for image i
                    norm_keypoints = target["keypoints"]  # Normalized keypoints for image i
                    labels.append(target["labels"])  # Labels
                    workout_labels.append(target["workout_labels"])  # Workout

                    image_width, image_height = image_sizes[i]

                    denorm_boxes = norm_boxes.clone()  # Clone the original normalized boxes
                    denorm_boxes[:, [0, 2]] = denorm_boxes[:, [0, 2]] * image_width  # Denormalize xmin, xmax
                    denorm_boxes[:, [1, 3]] = denorm_boxes[:, [1, 3]] * image_height  # Denormalize ymin, ymax)
                    denormalized_bbox.append(denorm_boxes)

                    denorm_keypoints = norm_keypoints.clone()  # Clone the original normalized keypoints
                    denorm_keypoints[..., 0] = denorm_keypoints[..., 0] * image_width  # Denormalize x (multiply by width)
                    denorm_keypoints[..., 1] = denorm_keypoints[..., 1] * image_height  # Denormalize y (multiply by height)
                    # visibility binary expected
                    denorm_keypoints[..., 2] = torch.where(denorm_keypoints[..., 2] < 0.3, 0, 1)
                    denormalized_keypoints.append(denorm_keypoints)

                # After denormalizing the boxes and keypoints, create the new targets
                new_targets = []

                for i in range(len(targets)):
                    new_targets.append({
                        "boxes": denormalized_bbox[i].to(targets[0]["boxes"].device),  # Shape: [1, 4]
                        "workout_labels": workout_labels[i].to(targets[0]["boxes"].device),  # Workout label for image i
                        "keypoints": denormalized_keypoints[i].to(targets[0]["boxes"].device),  # Shape: [1, 17, 3]
                        "labels": labels[i].to(targets[0]["boxes"].device),  # 0 background, 1 person
                    })


                features = self.model.backbone(x)  # Extract features from the backbone
                proposals, _ = self.model.rpn(image_list, features, new_targets)  # Region proposal network
                losses = self.model.roi_heads(features, proposals, image_sizes, new_targets)  # ROI heads output
                loss_dict = losses[1] # Index 0 is the output, index 1 is the loss 

                # Last feature map through the workout classification head
                final_feature_map = features['3']
                workout_label = self.workout_label_head(final_feature_map) 

                workout_label_loss = self.compute_losses(workout_label, new_targets)
                return loss_dict['loss_box_reg'], loss_dict['loss_keypoint'], workout_label_loss

            else: # Inference mode

                image_list = ImageList(x, [(image.shape[1], image.shape[2]) for image in x])
                image_sizes = image_list.image_sizes 

                features = self.model.backbone(x)
                proposals, _ = self.model.rpn(image_list, features)
                results = self.model.roi_heads(features, proposals, image_sizes)
                results_list = results[0] # Index 0 is the output, index 1 is the loss
                
                # Get most probable keypoints and boxes
                most_prob_boxes = []
                most_prob_keypoints = []
                for i in range(len(results_list)):
                    batch_idx_results = results_list[i]
                    highest_score_index = torch.argmax(batch_idx_results['scores']).item()

                    most_prob_boxes.append(batch_idx_results['boxes'][highest_score_index])
                    most_prob_keypoints.append(batch_idx_results['keypoints'][highest_score_index])

                # Convert lists to tensors after the loop
                bbox = torch.stack(most_prob_boxes)  # Assuming the boxes have the same shape
                keypoints = torch.stack(most_prob_keypoints)  # Assuming the keypoints have the same shape

                # Normalize bounding boxes (xmin, ymin, xmax, ymax) to [0, 1] based on image size
                normalized_bbox = []
                for idx, box in enumerate(bbox):
                    image_width, image_height = image_sizes[idx]
                    norm_box = [
                        box[0] / image_width,  # xmin
                        box[1] / image_height, # ymin
                        box[2] / image_width,  # xmax
                        box[3] / image_height  # ymax
                    ]
                    normalized_bbox.append(norm_box)
                normalized_bbox = torch.tensor(normalized_bbox, device=bbox.device)

                # Normalize keypoints (x, y) using the image size and bounding box width/height
                normalized_keypoints = []
                for idx, keypoint_set in enumerate(keypoints):
                    image_width, image_height = image_sizes[idx]
                    norm_keypoints = []
                    for keypoint in keypoint_set:
                        x, y, conf = keypoint[0], keypoint[1], keypoint[2]
                        # Normalize using the width and height of the bounding box
                        norm_x = x / image_width  # Normalize x by image width
                        norm_y = y / image_height  # Normalize y by image height
                        norm_keypoints.append([norm_x, norm_y, conf])
                    normalized_keypoints.append(norm_keypoints)
                normalized_keypoints = torch.tensor(normalized_keypoints, device=keypoints.device)

                # Last feature map through the workout classification head
                final_feature_map =  features['3']
                workout_label = self.workout_label_head(final_feature_map) 

                return normalized_bbox, normalized_keypoints, workout_label
    
    def compute_losses(self, outputs, targets, val=False):

        if self.model_label == 'resnet50' or val==True:
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
            visibility_mask = (confidence_scores > 0.3).float()  # If confidence > 0.3, include in loss

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
        
        elif self.model_label == 'keypoint-rcnn':
            # We only calculate the loss for the workout classifier because we will use the build-in loss for the keypoints and bbox

            workout_label_targets = torch.stack([target['workout_labels'] for target in targets]) 

            # Compute the workout label loss
            workout_label_indices = workout_label_targets.argmax(dim=1)  # Get the index of the target label
            workout_label_loss = nn.CrossEntropyLoss()(outputs, workout_label_indices)

            return workout_label_loss

