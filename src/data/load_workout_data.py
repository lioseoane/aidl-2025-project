import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def load_workout_data():
    # Hardcoded paths
    image_dir = "workout_dataset\images"
    annotation_dir = "workout_dataset\labels"

    # Initialize arrays to store the results
    keypoints_array = []
    images_array = []
    bounding_boxes_array = []
    classes_array = []

    # Loop through class subdirectories in the images directory
    for class_name in os.listdir(image_dir):
        class_img_dir = os.path.join(image_dir, class_name)
        class_annotation_dir = os.path.join(annotation_dir, class_name)

        # Skip if annotation directory doesn't exist
        if not os.path.exists(class_annotation_dir):
            continue

        # Loop through images in the class subdirectory
        for img_filename in os.listdir(class_img_dir):
            if img_filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter image files
                img_path = os.path.join(class_img_dir, img_filename)
                txt_path = os.path.join(class_annotation_dir, os.path.splitext(img_filename)[0] + ".txt")

                # Check if the annotation file exists
                if not os.path.exists(txt_path):
                    # print(f"Warning: No annotation file found for {img_filename}, skipping.")
                    continue  # Skip this image if annotation is not found

                # Parse annotation
                with open(txt_path, 'r') as f:
                    # Read content of annotation file
                    content = f.readline().strip()

                    # If the annotation content is empty or does not contain enough values
                    if not content:
                        # print(f"Warning: Annotation file {txt_path} is empty, skipping.")
                        continue
                    
                    # Split the values and convert them to float
                    values = list(map(float, content.split()))

                    # Ensure there are enough values for bbox (4) and keypoints (remaining)
                    if len(values) < 5:  # At least 4 for bounding box and 1 for keypoints
                        # print(f"Warning: Invalid annotation format in {txt_path}, skipping.")
                        continue

                    # Extract bounding box parameters (x_center, y_center, width, height)
                    x_center, y_center, width, height = values[1], values[2], values[3], values[4]

                    # Convert to bounding box format: [x_min, y_min, x_max, y_max]
                    x_min = x_center - (width / 2)
                    y_min = y_center - (height / 2)
                    x_max = x_center + (width / 2)
                    y_max = y_center + (height / 2)
                    bbox = [x_min, y_min, x_max, y_max]

                    # Reshape keypoints into [x, y] (ignoring visibility)
                    keypoints = [[values[i], values[i+1]] for i in range(5, len(values), 2)]

                # Only now that the annotation is valid, append to arrays
                # Append class label (folder name) to classes_array
                classes_array.append(class_name)  # class_name is the folder name

                # Append bounding box and keypoints to the respective arrays
                bounding_boxes_array.append(bbox)
                keypoints_array.append(keypoints)

                # Add image filename (URL) to images array
                images_array.append(img_path)

    return keypoints_array, images_array, bounding_boxes_array, classes_array


def plot_image_with_annotations(image_path, bbox, keypoints, class_name):
    """
    Plots an image with its bounding box and keypoints.

    Args:
        image_path (str): Path to the image file.
        bbox (list): Bounding box coordinates in [x_min, y_min, x_max, y_max] format.
        keypoints (list): List of keypoints in [x, y] format.
        class_name (str): Class name of the image.
    """
    # Open the image using PIL
    image = Image.open(image_path)

    # Get the width and height of the image for de-normalization
    img_width, img_height = image.size

    # Denormalize the bounding box
    denormalized_bbox = [bbox[0] * img_width, bbox[1] * img_height, bbox[2] * img_width, bbox[3] * img_height]

    # Denormalize the keypoints
    denormalized_keypoints = [[kp[0] * img_width, kp[1] * img_height] for kp in keypoints]

    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 9))

    # Display the image
    ax.imshow(image)

    # Add the bounding box to the image
    rect = patches.Rectangle(
        (denormalized_bbox[0], denormalized_bbox[1]),  # (x1, y1)
        denormalized_bbox[2] - denormalized_bbox[0],  # width = x2 - x1
        denormalized_bbox[3] - denormalized_bbox[1],  # height = y2 - y1
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)

    # Add keypoints to the image
    for (x, y) in denormalized_keypoints:
        ax.scatter(x, y, c='g', s=50, marker='x')  # Keypoints in green

    # Set the title and show the plot
    ax.set_title(f"Image: {os.path.basename(image_path)}\nClass: {class_name}")
    plt.show()


if __name__ == "__main__":
    # Load the dataset
    keypoints_array, images_array, bounding_boxes_array, classes_array = load_workout_data()

    # Get the first image and its annotations
    image_path = images_array[0]
    bbox = bounding_boxes_array[0]
    keypoints = keypoints_array[0]
    class_name = classes_array[0]

    # Plot the first image with its bounding box and keypoints
    plot_image_with_annotations(image_path, bbox, keypoints, class_name)