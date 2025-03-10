import cv2
import os
import glob

def extract_frames_from_videos(root_video_folder, output_root_folder):
    # Iterate through all subdirectories in the root video folder
    for folder in os.listdir(root_video_folder):
        folder_path = os.path.join(root_video_folder, folder)
        
        if os.path.isdir(folder_path):  # Ensure it's a folder
            video_files = glob.glob(os.path.join(folder_path, "*.mp4"))  # Get all video files

            for video_file in video_files:
                video_name = os.path.splitext(os.path.basename(video_file))[0]  # Extract video name (without extension)
                
                # Define the output folder structure
                save_path = os.path.join(output_root_folder, folder, f"{folder}_{video_name}")
                os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

                # Open video
                cap = cv2.VideoCapture(video_file)
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break  # Exit when video ends

                    frame_filename = os.path.join(save_path, f"{folder}_{video_name}_{frame_count:03d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    frame_count += 1

                cap.release()
                print(f"Frames extracted for {video_name}, saved to {save_path}")

if __name__ == "__main__":
    root_video_folder = "workout_dataset/youtube_videos"  # Root directory where all subfolders with videos exist
    output_root_folder = "workout_dataset/youtube_images"  # Root directory to save extracted frames

    extract_frames_from_videos(root_video_folder, output_root_folder)
