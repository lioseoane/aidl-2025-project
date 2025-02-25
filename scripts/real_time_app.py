import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import cv2
from PIL import Image
import numpy as np
from src.models.baseline_heatmap import baseline_heatmap
from torchvision import transforms
from src.utils.heatmaps import extract_keypoints_with_confidence

# Load your model (replace with your actual model)

model = baseline_heatmap(num_classes=20, num_keypoints=17, backbone='resnet50')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

state_dict = torch.load('checkpoints/model_epoch_52.pth', map_location=device)
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
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to float32 and scale the pixel values to [0,1]
    img_rgb = img_rgb.astype(np.float32) / 255.0

    # Define the normalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Manually normalize the image
    img_norm = (img_rgb - mean) / std

    # Convert from HWC to CHW format and create a tensor
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)  # shape: [C, H, W]

    # Add batch dimension and move to device
    img_tensor = img_tensor.unsqueeze(0).to(device)
    img_tensor = img_tensor.float()

    with torch.no_grad():
        output = model(img_tensor)

    # Extract the predicted values from output (assuming output is a tuple)
    bbox_pred = output[0]  # This should be the predicted bounding boxes
    keypoints_pred = extract_keypoints_with_confidence(output[1])  # This should be the predicted keypoints
    print(keypoints_pred)
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

    filtered_keypoints = []
    if keypoints_pred.shape[-1] == 3:  # If the keypoints have x, y, confidence
            for i, point in enumerate(keypoints_pred):
                x, y, confidence = point
                # Check if the keypoint is visible and within the bounding box
                if confidence > 0.5 and x_min <= x <= x_max and y_min <= y <= y_max:
                    # Draw the keypoint
                    cv2.circle(frame, (int(x * size_x), int(y * size_y)), 5, (0, 0, 255), -1)  # Red dots for keypoints
                    # Draw the index number next to the keypoint
                    cv2.putText(frame, str(i), (int(x * size_x) + 10, int(y * size_y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # White text for index
                    filtered_keypoints.append((i, x, y))  # Store valid keypoints for skeleton drawing
                    
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
                    filtered_keypoints.append((i, x, y))  # Store valid keypoints for skeleton drawing

    # Draw the skeleton using only valid keypoints
    for pair in SKELETON:
        i, j = pair
        valid_points = {kp[0]: (kp[1], kp[2]) for kp in filtered_keypoints}
        if i in valid_points and j in valid_points:
            x1, y1 = valid_points[i]
            x2, y2 = valid_points[j]
            cv2.line(frame, (int(x1 * size_x), int(y1 * size_y)), 
                     (int(x2 * size_x), int(y2 * size_y)), (255, 0, 0), 2)  # Blue lines


    # Display the result
    probabilities = torch.softmax(workout_label_pred[0], dim=0) 
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_class_name = idx_to_class_name[str(predicted_class_idx)]  # Map index to class name

    # Display class and probability
    cv2.putText(frame, 'Workout Label:', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f'{predicted_class_name}', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f'{probabilities[predicted_class_idx].item():.2f}', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 23)
        
    cv2.imshow('Webcam', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()