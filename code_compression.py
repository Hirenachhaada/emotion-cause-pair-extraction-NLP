import os
import subprocess

# 🔧 Set your input folder path here (absolute or relative)
INPUT_FOLDER = './output_scenes'  # <- change this

# Supported video formats
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv')

def compress_video(input_path, output_path):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-crf', '28',
        '-preset', 'fast',
        '-acodec', 'aac',
        '-b:a', '128k',
        output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def compress_videos_in_folder(folder_path):
    output_dir = os.path.join(folder_path, 'output')
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(VIDEO_EXTENSIONS):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_dir, f"{filename}")
            print(f"Compressing: {filename} → {filename}")
            compress_video(input_path, output_path)

if __name__ == "__main__":
    if not os.path.isdir(INPUT_FOLDER):
        print("❌ Invalid folder path:", INPUT_FOLDER)
    else:
        compress_videos_in_folder(INPUT_FOLDER)
        print("✅ All videos compressed and saved in 'output' folder.")
