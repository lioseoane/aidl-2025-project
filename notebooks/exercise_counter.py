import time
import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob

# Load YOLOv8 Pose Model
MODEL = YOLO('yolov8x-pose.pt')

# Function to calculate angle between three points
def calculate_angle(a: list, b: list, c: list) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Function to create video from frames
def frames_to_video(frames_dir, output_filename="output_video.mp4", fps=30):
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_files:
        print("No frames found. Video creation aborted.")
        return

    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        out.write(frame)
    
    out.release()
    print(f"Video saved successfully: {output_filename}")

# Exercise Counter Class
class ExerciseCounter:
    def __init__(self, conf_threshold: float = 0.5):
        self.counter = 0
        self.stage = None
        self.conf_threshold = conf_threshold

    def _get_angle(self, keypoints, kp_confs, indices):
        if all(i < len(keypoints) and kp_confs[i] >= self.conf_threshold for i in indices):
            return calculate_angle(keypoints[indices[0]], keypoints[indices[1]], keypoints[indices[2]])
        return None

    def process_frame(self, frame, keypoints, kp_confs, bbox, exercise_type):
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

        # Draw keypoints
        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Draw skeleton
        skeleton_pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        for i, j in skeleton_pairs:
            if i >= len(keypoints) or j >= len(keypoints) or keypoints[i][0] == 0 or keypoints[i][1] == 0 or keypoints[j][0] == 0 or keypoints[j][1] == 0:
                continue  # Skip if any keypoint is out of frame
            if i < len(keypoints) and j < len(keypoints):
                cv2.line(frame, (int(keypoints[i][0]), int(keypoints[i][1])),
                         (int(keypoints[j][0]), int(keypoints[j][1])), (255, 0, 0), 2)
                
        cv2.putText(frame, f'Reps: {self.counter} | Stage: {self.stage}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return frame

# Main Processing Function
def main(exercise):
    video_path = f"/home/asala/Desktop/aidl-2025-project/notebooks/video_samples/{exercise}_sample.mp4"
    output_frames_dir = "/home/asala/Desktop/aidl-2025-project/notebooks/output_frames/"
    output_video_path = f"/home/asala/Desktop/aidl-2025-project/notebooks/{exercise}_processed.mp4"
    os.makedirs(output_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    counter = ExerciseCounter()
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = MODEL(frame)
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
    IMAGE_DIR = "/home/codespace/aidl-2025-project/notebooks/output_frames/"
    for file in glob.glob(os.path.join(IMAGE_DIR, "*.jpg")):
        os.remove(file)
    main("squat")