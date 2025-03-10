import yt_dlp

def download_video(video_url, save_path="."):
    ydl_opts = {
        'outtmpl': f"{save_path}/%(title)s.%(ext)s",
        'format': 'best',  # This avoids separate video/audio merging
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

if __name__ == "__main__":
    download_video(video_url='https://www.youtube.com/shorts/BdUo8-A2QcM', save_path='workout_dataset/youtube_videos')
