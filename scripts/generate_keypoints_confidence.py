import os
import cv2
import shutil
from ultralytics import YOLO

IMAGE_DIR = "../workout_dataset/images"   # Path to your images
NEW_IMAGE_DIR = "../workout_dataset/images/all"
LABEL_DIR = "../workout_dataset/new_labels"   # Where .txt annotation files will be saved
MODEL_PATH = "yolo11x-pose.pt"
CLASS_ID = 0                           # For YOLO, person is often '0' if single class

model = YOLO(MODEL_PATH)

def delete_files_in_directory(directory_path):
   try:
     with os.scandir(directory_path) as entries:
       for entry in entries:
         if entry.is_file():
            os.unlink(entry.path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

if __name__ == "__main__":
    os.makedirs(LABEL_DIR, exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png")
    image_folders = [f for f in os.listdir(IMAGE_DIR)]

    for i, folder in enumerate(image_folders):
        image_filenames = [f for f in os.listdir(os.path.join(IMAGE_DIR, folder))]
        label_category_folder = os.path.join(LABEL_DIR, folder)
        if os.path.exists(label_category_folder):
            shutil.rmtree(label_category_folder)
        os.mkdir(label_category_folder)
        for j, image_filename in enumerate(image_filenames):
            image_path = os.path.join(IMAGE_DIR, folder, image_filename)
            image = cv2.imread(image_path)

            prediction_folder = image_filename[:-4]
            model.predict(image, name=prediction_folder, max_det=1, save_txt=True, save_conf=True)
            src_label = os.path.join("runs/pose", prediction_folder, "labels/image0.txt")
            dest_label = os.path.join(LABEL_DIR, folder, prediction_folder + ".txt")
            if os.path.isfile(src_label):
                shutil.copyfile(src_label, dest_label)
