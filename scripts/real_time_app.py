import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import cv2
from PIL import Image
import numpy as np
from src.models.resnet_with_heads import resnet_with_heads
from torchvision import transforms

# Load your model (replace with your actual model)
model = resnet_with_heads(num_classes=22, num_keypoints=17, backbone='resnet50')
state_dict = torch.load('resnet50.pth', map_location=torch.device('cpu'))
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cv2.namedWindow('Live Prediction', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live Prediction', 480, 360)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (480, 360))
    
    # Make prediction
    bbox_pred, keypoints_pred, workout_label_pred = predict(frame)
    
    # Draw the bounding boxes (bbox_pred should be in (x_min, y_min, x_max, y_max) format)
    for bbox in bbox_pred:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (int(x_min * 480), int(y_min * 360)), (int(x_max * 480), int(y_max * 360)), (0, 255, 0), 2)

    # Draw the keypoints (keypoints_pred should be a tensor with shape [num_keypoints, 2] for x, y)
    keypoints_pred = keypoints_pred[0]
    for i, point in enumerate(keypoints_pred):
        x, y, visibility = point
        if visibility > 0:  # Only draw visible keypoints
            # Draw the keypoint
            cv2.circle(frame, (int(x * 480), int(y * 360)), 5, (0, 0, 255), -1)  # Red dots for keypoints
            # Draw the index number next to the keypoint
            cv2.putText(frame, str(i), (int(x * 480) + 10, int(y * 360) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # White text for index

    # Draw the skeleton by connecting the keypoints
    for pair in SKELETON:
        i, j = pair
        x1, y1, _ = keypoints_pred[i]
        x2, y2, _ = keypoints_pred[j]
        if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:  # Check if both keypoints are visible
            cv2.line(frame, (int(x1 * 480), int(y1 * 360)), (int(x2 * 480), int(y2 * 360)), (255, 0, 0), 2)  # Blue lines for skeleton

    # Display the result
    predicted_class = torch.argmax(workout_label_pred[0]).item()
    cv2.putText(frame, f'Workout Label: {predicted_class}', (10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Live Prediction', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()