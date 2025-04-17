import os
import subprocess
import glob

# === CONFIGURATION ===
VIDEO_FOLDER = "./output"               # Folder containing .mkv files
AUDIO_OUTPUT_FOLDER = "extracted_audio"
AUDIO_FORMAT = "mp3"               # You can change this to 'wav', 'aac', etc.

# === CREATE OUTPUT FOLDER ===
os.makedirs(AUDIO_OUTPUT_FOLDER, exist_ok=True)

# === PROCESS EACH VIDEO FILE ===
video_files = glob.glob(os.path.join(VIDEO_FOLDER, "*.mkv"))

for video_path in video_files:
    filename = os.path.basename(video_path)
    name_without_ext = os.path.splitext(filename)[0]
    output_audio_path = os.path.join(AUDIO_OUTPUT_FOLDER, f"{name_without_ext}.{AUDIO_FORMAT}")

    print(f"🎧 Extracting audio from: {filename} ➜ {output_audio_path}")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",  # Best audio quality
        "-map", "a",
        output_audio_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"❌ Failed to extract audio from {filename}: {result.stderr.decode()}")
    else:
        print(f"✅ Audio saved: {output_audio_path}")

print("\n✅ All audio extracted.")
