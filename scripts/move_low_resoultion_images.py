import os
import sys
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image

IMAGES_DIR = 'workout_dataset/new_images'
OUTPUT_DIR = 'workout_dataset/low_res_images_360'  # Destination folder

MAX_RESOLUTION = 360  # Max width and height threshold

if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all image paths
    image_paths = []
    for root, dirs, files in os.walk(IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))

    print(f"Total images found: {len(image_paths)}")

    moved_count = 0

    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            width, height = img.size
            img.close()  # Explicitly close the image file to avoid WinError 32

            if width <= MAX_RESOLUTION and height <= MAX_RESOLUTION:
                # Compute relative path to maintain folder structure
                relative_path = os.path.relpath(image_path, IMAGES_DIR)
                destination_path = os.path.join(OUTPUT_DIR, relative_path)

                # Create destination folder if it doesn't exist
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)

                # Move the file
                shutil.move(image_path, destination_path)

                print(f"✅ Moved: {image_path} --> {destination_path}")
                moved_count += 1

        except Exception as e:
            print(f"❌ Failed to process {image_path}: {e}")

    print(f"\n✅ Done! {moved_count} images moved to {OUTPUT_DIR}")
