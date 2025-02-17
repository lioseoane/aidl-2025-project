import os
import cv2
import shutil
from ultralytics import YOLO

IMAGE_DIR = "../workout_dataset/images"
NEW_IMAGE_DIR = "../workout_dataset/new_images/"
MODEL_PATH = "yolo11x.pt"
CLASS_ID = 0  # Assuming class 0 corresponds to "person"

model = YOLO(MODEL_PATH)

def get_person_detections(image):
    """
    Run YOLO on the image and return a sorted list of tuples: (area, (x1, y1, x2, y2))
    for detections where the detected class equals CLASS_ID.
    The list is sorted from largest to smallest area.
    """
    detections = []
    results = model(image, max_det=2)
    result = results[0]
    boxes = result.boxes  # Contains all detections

    for idx in range(len(boxes)):
        if int(boxes.cls[idx].item()) == CLASS_ID:
            x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            area = (x2 - x1) * (y2 - y1)
            detections.append((area, (x1, y1, x2, y2)))
    
    # Sort detections from largest area to smallest
    detections.sort(key=lambda det: det[0], reverse=True)
    return detections

if __name__ == "__main__":
    os.makedirs(NEW_IMAGE_DIR, exist_ok=True)

    image_folders = [f for f in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, f))]
    for folder in image_folders:
        folder_path = os.path.join(IMAGE_DIR, folder)
        image_filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        new_folder_path = os.path.join(NEW_IMAGE_DIR, folder)
        if os.path.exists(new_folder_path):
            shutil.rmtree(new_folder_path)
        os.mkdir(new_folder_path)
        
        for image_filename in image_filenames:
            image_path = os.path.join(folder_path, image_filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Could not read image: {image_path}")
                continue

            detections = get_person_detections(image)
            if not detections:
                print(f"[INFO] No person detected in image: {image_path}")
                # Optionally, save the original image if no person detected
                cv2.imwrite(os.path.join(new_folder_path, image_filename), image)
                continue

            # The first detection is the largest one
            largest_detection = detections[0][1]
            print(f"[DEBUG] Image {image_filename}: Found {len(detections)} persons, largest box: {largest_detection}")

            output_image = image.copy()

            # Cover all detections except the largest one
            for _, box in detections[1:]:
                x1, y1, x2, y2 = box
                # Make sure the coordinates are within the image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, output_image.shape[1])
                y2 = min(y2, output_image.shape[0])
                output_image[y1:y2, x1:x2] = 0  # Black out the region

            new_image_path = os.path.join(new_folder_path, image_filename)
            cv2.imwrite(new_image_path, output_image)
