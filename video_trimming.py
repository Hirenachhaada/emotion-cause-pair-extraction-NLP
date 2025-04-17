import pandas as pd
import os
import subprocess
import re
import glob

# Constants (set these paths properly)
CSV_PATH = "train1.csv"
VIDEO_FOLDER = "./S3"  # Folder with your .mkv files
OUTPUT_FOLDER = "output_scenes"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Helper to extract start and end times
def parse_time_range(info):
    match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-\s*(\d{2}:\d{2}:\d{2}\.\d{3})', info)
    return match.group(1), match.group(2) if match else (None, None)

# Convert "Friends_S3E1" to "3x01"
def episode_to_file_prefix(ep):
    match = re.search(r'S(\d+)E(\d+)', ep)
    if match:
        season = int(match.group(1))
        episode = int(match.group(2))
        return f"{season}x{episode:02d}"
    return None

# Load CSV
df = pd.read_csv(CSV_PATH)

# Extract episode and times
df[['episode', 'time_range']] = df['episode_info'].str.split(': ', expand=True)
df[['start_time', 'end_time']] = df['time_range'].apply(lambda x: pd.Series(parse_time_range(x)))

# Group by scene
for (scene_id, episode), group in df.groupby(['scene_id', 'episode']):
    min_start = group['start_time'].min()
    max_end = group['end_time'].max()

    prefix = episode_to_file_prefix(episode)
    if prefix is None:
        print(f"⚠️ Could not parse episode name: {episode}")
        continue

    # Search for matching file (e.g., "3x01 - *.mkv")
    search_pattern = os.path.join(VIDEO_FOLDER, f"{prefix} - *.mkv")
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        print(f"🚫 No video found for {prefix}")
        continue

    video_path = matching_files[0]  # Take the first match
    scene_output = os.path.join(OUTPUT_FOLDER, f"{prefix}_scene{scene_id}.mkv")

    print(f"🎬 Trimming {video_path} Scene {scene_id}: {min_start} - {max_end}")
    command = [
        "ffmpeg",
        "-ss", min_start,
        "-to", max_end,
        "-i", video_path,
        "-c", 
        "copy",
        scene_output
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

print("✅ Scene trimming complete.")
