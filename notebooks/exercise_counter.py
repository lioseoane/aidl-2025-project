import time
import cv2
import numpy as np
from ultralytics import YOLO
from google.colab.patches import cv2_imshow


MODEL = YOLO('yolov8x-pose.pt')

def calculate_position(angle: float, key_body_part: str) -> str:
    """
    Determine the current position ("start", "medium", or "end") based on the
    calculated angle and the key body part.

    Parameters:
        angle (float): The computed joint angle.
        key_body_part (str): One of "upper body", "lower body", or "whole body".

    Returns:
        str: "start", "medium", or "end"
    """
    # Define example thresholds for each key body part.
    thresholds = {
        "upper body": {"start": 120, "end": 100},    # e.g., for push-ups or bench press
        "lower body": {"start": 150, "end": 100},     # e.g., for squats or deadlifts
        "whole body": {"start": 130, "end": 110}      # if needed
    }

    if key_body_part not in thresholds:
        raise ValueError(f"Unsupported key_body_part: {key_body_part}")

    start_th = thresholds[key_body_part]["start"]
    end_th   = thresholds[key_body_part]["end"]

    if angle >= start_th:
        return "start"
    elif angle <= end_th:
        return "end"
    else:
        return "medium"

def calculate_angle(a: list, b: list, c: list) -> float:
    """
    Calculate the angle (in degrees) at point b formed by the segments ab and cb.

    Parameters:
        a, b, c (list): Coordinates of the points [x, y].

    Returns:
        float: The angle in degrees.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

class ExerciseCounter:
    def __init__(self, conf_threshold: float = 0.5):
        self.counter: int = 0
        self.stage: str = None  # Represents the current position ("start", "medium", or "end")
        self.conf_threshold = conf_threshold

    def _get_angle(self, keypoints: np.ndarray, kp_confs: np.ndarray, indices: tuple) -> float:
        """
        Compute an angle only if all keypoints specified by the indices have sufficient confidence.

        Parameters:
            keypoints (np.ndarray): Array of keypoint coordinates.
            kp_confs (np.ndarray): Array of confidence scores for each keypoint.
            indices (tuple): A triplet of indices for angle calculation.

        Returns:
            float or None: The angle in degrees if valid; otherwise, None.
        """
        if all(kp_confs[i] >= self.conf_threshold for i in indices):
            return calculate_angle(keypoints[indices[0]], keypoints[indices[1]], keypoints[indices[2]])
        else:
            return None

    def _process_counter_exercise(self, frame: np.ndarray, keypoints: np.ndarray, kp_confs: np.ndarray,
                                  bbox, left_indices: tuple, right_indices: tuple, key_body_part: str) -> np.ndarray:
        """
        Unified processing for counter exercises.

        Parameters:
            frame (np.ndarray): The current video frame.
            keypoints (np.ndarray): Detected pose keypoints.
            kp_confs (np.ndarray): Confidence scores for keypoints.
            bbox: Bounding box of the detected person.
            left_indices (tuple): Keypoint indices for one side (e.g. left side).
            right_indices (tuple): Keypoint indices for the opposite side.
            key_body_part (str): Specifies which body segment is being used ("upper body" or "lower body").

        Returns:
            np.ndarray: The frame with overlay (bounding box and counter).
        """
        left_angle = self._get_angle(keypoints, kp_confs, left_indices)
        right_angle = self._get_angle(keypoints, kp_confs, right_indices)
        print(f"Angle: Left: {left_angle}, Right: {right_angle}")

        # Use available angle(s)
        if left_angle is not None and right_angle is not None:
            angle = (left_angle + right_angle) / 2.0
        elif left_angle is not None:
            angle = left_angle
        elif right_angle is not None:
            angle = right_angle
        else:
            return frame  # Skip frame if no valid angle

        # Get the current position using our unified calculate_position function.
        new_position = calculate_position(angle, key_body_part)
        if new_position == "medium":
            new_position = self.stage
        print(f"{key_body_part.capitalize()} angle: {angle:.1f} -> Position: {new_position}")

        # Count a rep when a full cycle is detected: transition from "end" to "start".
        if self.stage == "end" and new_position == "start":
            self.counter += 1
        self.stage = new_position

        # Draw a bounding box and overlay the counter.
        cv2.rectangle(frame,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      (0, 255, 0), 2)
        cv2.putText(frame, f'Reps: {self.counter}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame

    # For counter–based exercises, we now simply call _process_counter_exercise with the appropriate parameters.
    def _process_pushup(self, frame: np.ndarray, keypoints: np.ndarray, kp_confs: np.ndarray, bbox) -> np.ndarray:
        return self._process_counter_exercise(frame, keypoints, kp_confs, bbox,
                                                left_indices=(5, 7, 9), right_indices=(6, 8, 10),
                                                key_body_part="upper body")

    def _process_bench_press(self, frame: np.ndarray, keypoints: np.ndarray, kp_confs: np.ndarray, bbox) -> np.ndarray:
        return self._process_counter_exercise(frame, keypoints, kp_confs, bbox,
                                                left_indices=(5, 7, 9), right_indices=(6, 8, 10),
                                                key_body_part="upper body")

    def _process_squat(self, frame: np.ndarray, keypoints: np.ndarray, kp_confs: np.ndarray, bbox) -> np.ndarray:
        return self._process_counter_exercise(frame, keypoints, kp_confs, bbox,
                                                left_indices=(11, 13, 15), right_indices=(12, 14, 16),
                                                key_body_part="lower body")

    def _process_deadlift(self, frame: np.ndarray, keypoints: np.ndarray, kp_confs: np.ndarray, bbox) -> np.ndarray:
        return self._process_counter_exercise(frame, keypoints, kp_confs, bbox,
                                                left_indices=(11, 13, 15), right_indices=(12, 14, 16),
                                                key_body_part="whole body")

    def process_frame(self, frame: np.ndarray, keypoints: np.ndarray, kp_confs: np.ndarray,
                      bbox, exercise_type: str) -> np.ndarray:
        """
        Dispatch processing to the appropriate exercise method.
        """
        exercise_funcs = {
            "push-up": self._process_pushup,
            "benchpress": self._process_bench_press,
            "squat": self._process_squat,
            "deadlift": self._process_deadlift
        }
        if exercise_type in exercise_funcs:
            return exercise_funcs[exercise_type](frame, keypoints, kp_confs, bbox)
        else:
            return frame

def main(exercise: str):
    video_path = f"/content/drive/MyDrive/data/human_pose_exercise/{exercise}_sample.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    counter = ExerciseCounter(conf_threshold=0.5)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print("Warning: Could not determine FPS, defaulting to 30")
        video_fps = 30
    print(f"Video FPS: {video_fps}")
    # For rep-based exercises, process every frame or use a small skip.
    skip_frames = 4 if exercise in ["squat", "push-up", "benchpress", "deadlift"] else int(video_fps)
    print(f"Skipping every {skip_frames} frames for {exercise}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % skip_frames == 0:
            result = MODEL(frame)
            boxes = result[0].boxes.xyxy.cpu().numpy() if result[0].boxes else []
            print(boxes)
            if result[0].keypoints is not None:
                keypoints = result[0].keypoints.xy.cpu().numpy()[0]
                kp_confs = result[0].keypoints.conf.cpu().numpy()[0]
                print("Keypoint confidences:", kp_confs)
            else:
                keypoints, kp_confs = None, None

            if boxes is not None and len(boxes) > 0 and keypoints is not None and kp_confs is not None:
                bbox = boxes[0]
                frame = counter.process_frame(frame, keypoints, kp_confs, bbox, exercise)
                # Optionally display the full plotted result
                img_full = result[0].plot()
                height, width = img_full.shape[:2]
                img_resized = cv2.resize(img_full, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                cv2_imshow(img_resized)

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final {exercise} count: {counter.counter}")

if __name__ == "__main__":
    # Choose an exercise: "push-up", "benchpress", "squat" or "deadlift"
    exercise_type = "deadlift"
    main(exercise_type)
