import os
import cv2
from ultralytics import YOLO

IMAGE_DIR = "../workout_dataset/images"   # Path to your images
LABEL_DIR = "../workout_dataset/labels"   # Where .txt annotation files will be saved
MODEL_PATH = "yolo11x-pose.pt"
CLASS_ID = 0                           # For YOLO, person is often '0' if single class

model = YOLO(MODEL_PATH)

if __name__ == "__main__":
    os.makedirs(LABEL_DIR, exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png")
    image_folders = [f for f in os.listdir(IMAGE_DIR)]
    # image_files.sort()

    for i, folder in enumerate(image_folders):
        image_filenames = [f for f in os.listdir(os.path.join(IMAGE_DIR, folder))]
        label_filenames = [f for f in os.listdir(os.path.join(LABEL_DIR, folder))]
        for j, image_filename in enumerate(image_filenames):
            image_path = os.path.join(IMAGE_DIR, folder, image_filename)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            results = model(image)

            result = results[0]
            boxes = result.boxes.xywhn
            keypoints = result.keypoints.xyn
            for k, keypoint in enumerate(keypoints):
                # print(f"Keypoint {k}", keypoint)
                label_path = os.path.join(LABEL_DIR, folder, image_filename[:-4] + ".txt")
                print(label_path)
                with open(label_path, "w") as f:
                    for det_i in range(len(boxes)):
                        box = boxes[det_i].tolist()  # [cx, cy, w, h]
                        kp_array = keypoints[det_i]
                        
                        if len(kp_array) > 10:
                            line_parts = []
                            line_parts.append("0")
                            line_parts.extend([f"{x:.6f}" for x in box])

                            for (kp_x, kp_y) in kp_array:
                                # we reuse the confidence as "visibility" or you can set a fixed 2
                                line_parts.extend([f"{kp_x:.6f}", f"{kp_y:.6f}"])

                            line_str = " ".join(line_parts)
                            f.write(line_str + "\n")

            # Show image
            # annotated_image = result.plot()
            # cv2.imshow("Pose", annotated_image)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break
