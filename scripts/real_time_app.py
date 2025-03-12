import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import cv2
import numpy as np
from src.models.heatmap_fpn_v3 import heatmap_fpn
from src.utils.heatmaps import extract_keypoints_with_confidence, extract_bbox_from_heatmaps

# Kalmar fiter --> Get smoothed keypoints
class KalmanFilterKeypoint:
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)  # State: [x, y, dx, dy] | Measurement: [x, y]

        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + dx
            [0, 1, 0, 1],  # y = y + dy
            [0, 0, 1, 0],  # dx remains dx
            [0, 0, 0, 1]   # dy remains dy
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)

    def update(self, x, y):
        measurement = np.array([[x], [y]], dtype=np.float32)
        if np.all(self.kf.statePost[:2] == 0):  # If first frame, initialize state
            self.kf.statePost[:2] = measurement
        self.kf.correct(measurement)

    def predict(self):
        predicted_state = self.kf.predict()
        return predicted_state[0][0], predicted_state[1][0]  # Predicted (x, y)


# Initialize model and load checkpoint
model = heatmap_fpn(num_classes=20, num_keypoints=17, backbone='resnet50')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
state_dict = torch.load('checkpoints/model_epoch_34.pth', map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Skeleton connections
SKELETON = [
    (0, 1), (1, 2), (2, 0),     # Nose -> Left Eye -> Right Eye -> Nose,
    (1, 3),                     # Left Eye -> Left Ear
    (2, 4),                     # Right Eye -> Right Ear
    (5, 6),                     # Left Shoulder -> Right Shoulder
    (5, 7), (7, 9),             # Left Shoulder -> Left Elbow -> Left Wrist
    (6, 8), (8, 10),            # Right Shoulder -> Right Elbow -> Right Wrist
    (5, 11), (6, 12),           # Left Shoulder -> Left Hip, Right Shoulder -> Right Hip
    (11, 13), (13, 15),         # Left Hip -> Left Knee -> Left Ankle
    (12, 14), (14, 16)          # Right Hip -> Right Knee -> Right Ankle
]

keypoint_mapping = ['Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Year', 'Left Shoulder', 'Right Shoulder',
                    'Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip', 'Left Knee',
                    'Right Knee', 'Left Ankle', 'Right Ankle']

# Initialize Kalman filters for all keypoints
num_keypoints = 17
kalman_filters = [KalmanFilterKeypoint() for _ in range(num_keypoints)]

# Confidence threshold for keypoints
confidence_threshold = 0.5

# predict the model
def predict(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    target_w, target_h = [224, 224]

    scale = min(target_w / float(w), target_h / float(h))
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h))

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    sample_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize
    sample_tensor = sample_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sample_tensor)

    bbox_pred = extract_bbox_from_heatmaps(output[0])
    keypoints_pred = extract_keypoints_with_confidence(output[1], refine=False)
    workout_label_pred = output[2]  
    heatmaps_pred = output[1].cpu().numpy() 
    bbox_heatmaps_pred = output[0].cpu().numpy()

    return bbox_pred, keypoints_pred, workout_label_pred, heatmaps_pred, bbox_heatmaps_pred


# Open webcam
cap = cv2.VideoCapture(0)

assert cap.isOpened(), "Error: Could not open webcam."

# Size of the App
size_x, size_y = 448, 448
cap.set(cv2.CAP_PROP_FRAME_WIDTH, size_x)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size_y)
cv2.namedWindow('Live Prediction', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live Prediction', size_x, size_y)

# Load idx_to_class_name during inference
#with open('idx_to_class_name.json', 'r') as f:
    #idx_to_class_name = json.load(f)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (size_x, size_y))
        
    # Make prediction
    bbox_pred, keypoints_pred, workout_label_pred, heatmaps_pred, bbox_heatmaps_pred = predict(frame)

    # Display keypoints heatmap
    combined_heatmap = np.sum(heatmaps_pred[0], axis=0)  # Average across all keypoints
    combined_heatmap = cv2.resize(combined_heatmap, (size_x, size_y))  # Resize to match the frame size
    combined_heatmap = cv2.normalize(combined_heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # Normalize to 0-255
    heatmap_colored = cv2.applyColorMap(combined_heatmap, cv2.COLORMAP_JET)  # Apply color map
    cv2.imshow('Predicted Heatmap', heatmap_colored)

    # Display bbox heatmap
    bbox_heatmap_resized = cv2.resize(bbox_heatmaps_pred[0], (size_x, size_y))  
    bbox_heatmap_normalized = cv2.normalize(bbox_heatmap_resized, None, 0, 255, cv2.NORM_MINMAX , dtype=cv2.CV_8U)
    bbox_heatmap_colored = cv2.applyColorMap(bbox_heatmap_normalized, cv2.COLORMAP_JET)
    cv2.imshow('BBox Heatmap', bbox_heatmap_colored)

    # Draw the bounding boxes
    for bbox in bbox_pred:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (int(x_min * size_x), int(y_min * size_y)), (int(x_max * size_x), int(y_max * size_y)), (0, 255, 0), 1)

    # Draw the keypoints
    keypoints_pred = keypoints_pred[0]

    filtered_keypoints = []
    for i, point in enumerate(keypoints_pred):
        x, y, confidence = point

        if confidence >= confidence_threshold:
            x, y = x.item(), y.item()
            kalman_filters[i].update(x, y)  # Update Kalman filter
            x, y = kalman_filters[i].predict()  # Get smoothed keypoints
            cv2.circle(frame, (int(x * size_x), int(y * size_y)), 1, (0, 0, 255), -1)  # Red dots for keypoints
            if i == 0:
                padding_x = 0
                padding_y = 10
            elif i % 2 == 0:
                padding_x = -10
                adding_y = -10
            else:
                padding_x = 10
                padding_y = -10
            cv2.putText(frame, keypoint_mapping[i], (int(x * size_x) + padding_x, int(y * size_y) + padding_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA) 
            filtered_keypoints.append((i, x, y))  # Store valid keypoints for skeleton drawing

    # Draw the skeleton 
    for pair in SKELETON:
        i, j = pair
        valid_points = {kp[0]: (kp[1], kp[2]) for kp in filtered_keypoints}
        if i in valid_points and j in valid_points:
            x1, y1 = valid_points[i]
            x2, y2 = valid_points[j]
            cv2.line(frame, (int(x1 * size_x), int(y1 * size_y)), 
                    (int(x2 * size_x), int(y2 * size_y)), (0, 0, 255), 1)  # Blue lines


    # Display workout class and probability
    #probabilities = torch.softmax(workout_label_pred[0], dim=0) 
    #predicted_class_idx = torch.argmax(probabilities).item()
    #predicted_class_name = idx_to_class_name[str(predicted_class_idx)]  # Map index to class name

    #if probabilities[predicted_class_idx].item() > 0.75:   
        #colour = (0, 255, 0)
    #else:
        #colour = (0, 0, 255)

    #cv2.putText(frame, 'Workout Label:', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    #cv2.putText(frame, f'{predicted_class_name}', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    #cv2.putText(frame, f'{probabilities[predicted_class_idx].item():.2f}', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
            
    # Display the frame
    cv2.imshow('Live Prediction', frame)
        
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()