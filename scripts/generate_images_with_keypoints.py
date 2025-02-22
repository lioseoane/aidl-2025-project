import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from src.data.load_workout_data import load_workout_data
from src.utils.visualization import visualize_keypoints
import numpy as np

IMAGES_DIR = 'workout_dataset/new_images'
KEYPOINTS_DIR = 'workout_dataset/new_labels'
OUTPUT_DIR = 'workout_dataset/new_images_labels'

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    keypoints_array, images_array, bounding_boxes_array, classes_array = load_workout_data()

    for i in range(len(keypoints_array)):
         
        image = Image.open(images_array[i]).convert("RGB")

        w, h = image.size

        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))

        keypoints = np.array(keypoints_array[i], dtype=np.float32)
        bbox = np.array(bounding_boxes_array[i], dtype=np.float32)

        image_with_keypoints = visualize_keypoints(image, keypoints, keypoints, w, h, bbox, bbox)
        image_with_keypoints = np.transpose(image_with_keypoints, (1, 2, 0)) 

        output_image = Image.fromarray(image_with_keypoints)

        relative_path = os.path.relpath(images_array[i], "workout_dataset/new_images") 
        folder, filename = os.path.split(relative_path) 
        filename_without_ext = os.path.splitext(filename)[0]  

        output_path = f'{OUTPUT_DIR}/{folder}/{filename}.jpg'
        os.makedirs(f'{OUTPUT_DIR}/{folder}', exist_ok=True)
        output_image.save(f'{output_path}')
