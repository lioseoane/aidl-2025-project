import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import cv2
from PIL import Image
import numpy as np
from src.models.resnet_with_heads import resnet_with_heads
from torchvision import transforms

# Load your model (replace with your actual model)
model = resnet_with_heads(num_classes=22, num_keypoints=17, backbone='resnet50')
state_dict = torch.load('checkpoints/model_epoch_30.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Skeleton structure for YOLO 17 keypoints (pairs of keypoints to connect)
SKELETON = [
    (0, 1), (1, 2), (2, 3), (2, 4),  # Nose -> Left Eye -> Right Eye, Right Eye -> Right Ear, Left Eye -> Left Ear
    (5, 6),                          # Left Shoulder -> Right Shoulder
    (5, 7), (7, 9),                  # Left Shoulder -> Left Elbow -> Left Wrist
    (6, 8), (8, 10),                 # Right Shoulder -> Right Elbow -> Right Wrist
    (5, 11), (6, 12),                # Left Shoulder -> Left Hip, Right Shoulder -> Right Hip
    (11, 13), (13, 15),              # Left Hip -> Left Knee -> Left Ankle
    (12, 14), (14, 16)               # Right Hip -> Right Knee -> Right Ankle
]

def predict(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Define transformation to convert PIL image to Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor (automatically normalizes to [0,1])
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transform and add batch dimension
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)

    # Extract the predicted values from output (assuming output is a tuple)
    bbox_pred = output[0]  # This should be the predicted bounding boxes
    keypoints_pred = output[1]  # This should be the predicted keypoints
    workout_label_pred = output[2]  # This should be the predicted workout label

    return bbox_pred, keypoints_pred, workout_label_pred


# Open webcam
cap = cv2.VideoCapture(0)

# Size of the App
size_x, size_y = 224, 224

cap.set(cv2.CAP_PROP_FRAME_WIDTH, size_x)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size_y)
cv2.namedWindow('Live Prediction', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live Prediction', size_x, size_y)

# Load idx_to_class_name during inference
with open('idx_to_class_name.json', 'r') as f:
    idx_to_class_name = json.load(f)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (size_x, size_y))
    
    # Make prediction
    bbox_pred, keypoints_pred, workout_label_pred = predict(frame)
    
    # Draw the bounding boxes (bbox_pred should be in (x_min, y_min, x_max, y_max) format)
    for bbox in bbox_pred:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (int(x_min * size_x), int(y_min * size_y)), (int(x_max * size_x), int(y_max * size_y)), (0, 255, 0), 2)

    # Draw the keypoints (keypoints_pred should be a tensor with shape [num_keypoints, 2] for x, y)
    keypoints_pred = keypoints_pred[0]

    if keypoints_pred.shape[-1] == 3:  # If the keypoints have x, y, visibility
            for i, point in enumerate(keypoints_pred):
                x, y, visibility = point
                # Check if the keypoint is visible and within the bounding box
                if visibility > 0 and x_min <= x <= x_max and y_min <= y <= y_max:
                    # Draw the keypoint
                    cv2.circle(frame, (int(x * size_x), int(y * size_y)), 5, (0, 0, 255), -1)  # Red dots for keypoints
                    # Draw the index number next to the keypoint
                    cv2.putText(frame, str(i), (int(x * size_x) + 10, int(y * size_y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # White text for index
                    
    elif keypoints_pred.shape[-1] == 2:  # If the keypoints have only x, y 
        for i, point in enumerate(keypoints_pred):
                x, y = point
                # Check if the keypoint is within the bounding box
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    # Draw the keypoint
                    cv2.circle(frame, (int(x * size_x), int(y * size_y)), 5, (0, 0, 255), -1)  # Red dots for keypoints
                    # Draw the index number next to the keypoint
                    cv2.putText(frame, str(i), (int(x * size_x) + 10, int(y * size_y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # White text for index

    # Draw the skeleton by connecting the keypoints
    for pair in SKELETON:
        i, j = pair
            # Check if keypoints have 3 values (x, y, visibility) or just 2 (x, y)
        if keypoints_pred.shape[-1] == 3:
            x1, y1, _ = keypoints_pred[i]  # Unpacking 3 values
            x2, y2, _ = keypoints_pred[j]
        elif keypoints_pred.shape[-1] == 2:
            x1, y1 = keypoints_pred[i]  # Unpacking 2 values
            x2, y2 = keypoints_pred[j]

        if (x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0 and
                x_min <= x1 <= x_max and y_min <= y1 <= y_max and
                x_min <= x2 <= x_max and y_min <= y2 <= y_max):
                cv2.line(frame, (int(x1 * size_x), int(y1 * size_y)), 
                         (int(x2 * size_x), int(y2 * size_y)), (255, 0, 0), 2)  # Blue lines for skeleton


    # Display the result
    probabilities = torch.softmax(workout_label_pred[0], dim=0) 
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_class_name = idx_to_class_name[str(predicted_class_idx)]  # Map index to class name

    # Display class and probability
    cv2.putText(frame, f'Workout Label: {predicted_class_name} ({probabilities[predicted_class_idx].item():.2f})', 
                (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Webcam', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()