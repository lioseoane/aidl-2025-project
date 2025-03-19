import sys
import os
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collections import deque
import time
import cv2
import numpy as np
import json
import torch
from ultralytics import YOLO

from src.models.heatmap_fpn_v3 import heatmap_fpn
from src.utils.heatmaps import extract_keypoints_with_confidence, extract_bbox_from_heatmaps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FRAME_SKIP = 10  # Process every 3rd frame for efficiency
CONF_HISTORY = 5  # Number of past classifications to consider
class_smoothing = deque(maxlen=CONF_HISTORY)

def load_model(our_model = False):
    if our_model:
        model = heatmap_fpn(num_classes=20, num_keypoints=17, backbone='resnet50')
        model.to(device)
        state_dict = torch.load('model_epoch_130.pth', map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    else:
        return YOLO('yolov8x-pose.pt')

# Function to calculate angle between three points
def calculate_angle(a: list, b: list, c: list) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Function to create video from frames
def frames_to_video(frames_dir, output_filename="output_video.mp4", fps=30, frame_size=(352, 352)):
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_files:
        print("No frames found. Video creation aborted.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is None:
            continue  # Skip invalid frames

        frame_resized = cv2.resize(frame, frame_size)  # Resize to match prediction size
        out.write(frame_resized)

    out.release()
    print(f"Video saved successfully: {output_filename}")

# predict the model
def predict(frame, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    target_w, target_h = [352, 352]

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
    keypoints_pred = extract_keypoints_with_confidence(output[1])

    with open('heatmap_fpn_v3.json', 'r') as f:
        idx_to_class_name = json.load(f)
    probabilities = torch.softmax(output[2][0], dim=0)
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_class_name = idx_to_class_name[str(predicted_class_idx)]
    class_smoothing.append(predicted_class_name)
    smoothed_class = max(set(class_smoothing), key=class_smoothing.count)

    return bbox_pred, keypoints_pred, smoothed_class, h, w, pad_left, pad_top, new_w, new_h

# Exercise Counter Class
class ExerciseCounter:
    def __init__(self, conf_threshold: float = 0.0):
        self.counter = 0
        self.stage = None
        self.conf_threshold = conf_threshold

    def _get_angle(self, keypoints, kp_confs, indices):
        if all(i < len(keypoints) and kp_confs[i] >= self.conf_threshold for i in indices):
            return calculate_angle(keypoints[indices[0]], keypoints[indices[1]], keypoints[indices[2]])
        return None

    def process_frame(self, frame, keypoints, kp_confs, bbox, exercise_type, h, w, pad_left, pad_top, new_w, new_h):
        left_indices, right_indices, key_body_part = {
            "deadlift": ((11, 13, 15), (12, 14, 16), "whole body"),
            "squat": ((11, 13, 15), (12, 14, 16), "lower body"),
            "push-up": ((5, 7, 9), (6, 8, 10), "upper body"),
            "benchpress": ((5, 7, 9), (6, 8, 10), "upper body")
        }.get(exercise_type, (None, None, None))

        if not left_indices:
            return frame

        left_angle = self._get_angle(keypoints, kp_confs, left_indices)
        right_angle = self._get_angle(keypoints, kp_confs, right_indices)
        angle = (left_angle + right_angle) / 2.0 if left_angle and right_angle else left_angle or right_angle
        if angle is None or any(i >= len(keypoints) for i in left_indices + right_indices):
            print("Skipping frame due to missing keypoints")
            print("angle is None: ", angle is None)
            print("any: ", any(i >= len(keypoints) for i in left_indices + right_indices))
            return frame

        thresholds = {
            "upper body": {"start": 120, "end": 100},
            "lower body": {"start": 150, "end": 100},
            "whole body": {"start": 130, "end": 110}
        }
        
        if key_body_part not in thresholds:
            raise ValueError(f"Unsupported key_body_part: {key_body_part}")

        start_th = thresholds[key_body_part]["start"]
        end_th = thresholds[key_body_part]["end"]

        if angle >= start_th:
            new_stage = "start"
        elif angle <= end_th:
            new_stage = "end"
        else:
            new_stage = self.stage
        if self.stage == "end" and new_stage == "start":
            self.counter += 1
        self.stage = new_stage
        # Ensure bbox is properly converted to a list
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy().tolist()
        # Draw bounding box
        if bbox is not None:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw skeleton and keypoints
        skeleton_pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        
        for i, j in skeleton_pairs:
            if i >= len(keypoints) or j >= len(keypoints) or keypoints[i][0] == 0 or keypoints[i][1] == 0 or keypoints[j][0] == 0 or keypoints[j][1] == 0:
                continue  # Skip if any keypoint is out of frame
            if i < len(keypoints) and j < len(keypoints):
                
                # Extract keypoints (normalized values 0-1)
                xi, yi = keypoints[i][0] * 352, keypoints[i][1] * 352
                xj, yj = keypoints[j][0] * 352, keypoints[j][1] * 352
                # Convert keypoints from resized (352x352) space to original frame
                xi = ((xi - pad_left) / new_w) * w
                yi = ((yi - pad_top) / new_h) * h
                xj = ((xj - pad_left) / new_w) * w
                yj = ((yj - pad_top) / new_h) * h
                xi, yi = int(xi), int(yi)
                xj, yj = int(xj), int(yj)


                cv2.circle(frame, (int(xi), int(yi)), 5, (0, 255, 0), -1)
                cv2.circle(frame, (int(xj), int(yj)), 5, (0, 255, 0), -1)
                cv2.line(frame, (xi, yi), (xj, yj), (255, 0, 0), 2)
                
        cv2.putText(frame, f'Exercise: {exercise_type}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        
        cv2.putText(frame, f'Reps: {self.counter} | Stage: {self.stage}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return frame

# Main Processing Function
def main(exercise, our_model = False):
    video_path = f"notebooks/video_samples/{exercise}_sample.mp4"
    output_frames_dir = "demo/output_frames/"
    output_video_path = f"demo/{exercise}_processed_0.0.mp4"
    os.makedirs(output_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counter = ExerciseCounter()
    frame_index = 0
    model = load_model(our_model)

    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            break
        if our_model:
            bbox, keypoints, exercise, h, w, pad_left, pad_top, new_w, new_h = predict(frame, model)
            kp_confs = []
            for i, point in enumerate(keypoints[0]):
                x, y, confidence = point
                kp_confs.append(confidence)
            frame = counter.process_frame(frame, keypoints[0], kp_confs, bbox[0], exercise, h, w, pad_left, pad_top, new_w, new_h)

        else:
            result = model(frame)
            # currenlty only processing in yolo format frame
            if result[0].keypoints is not None and len(result[0].keypoints.xy.cpu().numpy()[0]) >= 17:
                keypoints = result[0].keypoints.xy.cpu().numpy()[0]
                kp_confs = result[0].keypoints.conf.cpu().numpy()[0]
                bbox = result[0].boxes.xyxy.cpu().numpy()[0] if result[0].boxes else None
                frame = counter.process_frame(frame, keypoints, kp_confs, bbox, exercise)

        frame_filename = os.path.join(output_frames_dir, f"frame_{frame_index:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final {exercise} count: {counter.counter}")
    frames_to_video(output_frames_dir, output_video_path, fps)

if __name__ == "__main__":
    IMAGE_DIR = "notebooks/output_frames/"
    for file in glob.glob(os.path.join(IMAGE_DIR, "*.jpg")):
        os.remove(file)
    main("squat", our_model = True)
