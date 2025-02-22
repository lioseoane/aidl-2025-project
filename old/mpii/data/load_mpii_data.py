import scipy.io as sio
import numpy as np
import os

def load_mpii_data():
    # Path to the .mat file (adjust if necessary)
    mat_file = "data/mpii_human_pose/mpii_human_pose.mat"
    
    # Load the .mat file
    mat_data = sio.loadmat(mat_file, struct_as_record=False)
    release_data = mat_data['RELEASE']
    
    # Extract relevant data from the file
    obj_rel = release_data[0, 0]
    annolist = obj_rel.__dict__['annolist']
    img_train = obj_rel.__dict__['img_train']
    act = obj_rel.__dict__['act']
    
    # Lists to store images, keypoints, and head boxes
    keypoints_list = []
    images_list = []
    head_boxes_list = []
    
    # Loop through the annotations
    for ix in range(0, annolist.shape[1]):
        # Extract annotation for each image
        rect = annolist[0, ix].__dict__['annorect']
        
        if rect.shape[0] == 0:
            continue  # No annotation for this image
        
        obj_rect = rect[0, 0]
        
        # Extract head box
        head_box = []
        if 'x1' in obj_rect.__dict__ and 'y1' in obj_rect.__dict__ and 'x2' in obj_rect.__dict__ and 'y2' in obj_rect.__dict__:
            x1 = obj_rect.x1.item()
            y1 = obj_rect.y1.item()
            x2 = obj_rect.x2.item()
            y2 = obj_rect.y2.item()
            
            # Append head box to the list
            head_box.append([x1, y1, x2, y2])  # x1, y1, x2, y2
        else:
            continue  # Skip this image if no head box
        
        # Extract keypoints if available
        if 'annopoints' not in obj_rect.__dict__ or obj_rect.__dict__['annopoints'].shape[0] == 0:
            continue  # No keypoints for this image
        
        annopoints = obj_rect.__dict__['annopoints']
        obj_points = annopoints[0, 0]
        points = obj_points.__dict__['point']
        
        if points.shape[1] > 0:
            points_values = []
            for point in points[0, :]:
                x_value = point.x.item()  # Extract x-coordinate
                y_value = point.y.item()  # Extract y-coordinate

                points_values.append([x_value, y_value])  # Append point as [x, y]
            
            if len(points_values) == 16:
                # Append to the lists
                keypoints_list.append(points_values)
                head_boxes_list.append(head_box[0])  # We only store one head box for each image

                # Store corresponding image
                image = annolist[0, ix].__dict__['image'][0, 0].__dict__['name'][0]
                images_list.append(image)
    
    # Convert lists to numpy arrays
    keypoints_array = np.array(keypoints_list, dtype=np.float32)
    images_array = np.array(images_list, dtype=object)
    head_boxes_array = np.array(head_boxes_list, dtype=np.float32)

    # Debugging outputs
    # print(f"Total images: {images_array.shape[0]}")
    # print(f"Total images with keypoints: {keypoints_array.shape[0]}")
    # print(f"Total bounding boxes: {head_boxes_array.shape[0]}")
    
    return keypoints_array, images_array, head_boxes_array
