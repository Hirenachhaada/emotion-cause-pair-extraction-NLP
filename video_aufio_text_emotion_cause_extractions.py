import os
import subprocess
import whisper
import streamlit as st
import requests
import tempfile

# --- Extract Audio using FFmpeg ---
def extract_audio_ffmpeg(video_path, output_audio_path="temp_audio.wav"):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_audio_path

# --- Transcribe Audio ---
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# --- Actual API Calls to Local Dummy Server ---
def call_video_emotion_api(video_path):
    with open(video_path, 'rb') as f:
        files = {'file': f}
        response = requests.post("https://api.ngrok.com/tunnels/hidn54domwsoa15678/video_emotion", files=files)
    return response.json()

def call_audio_emotion_api(audio_path):
    with open(audio_path, 'rb') as f:
        files = {'file': f}
        response = requests.post("https://api.ngrok.com/tunnels/hidn54domwsoa15678/audio_emotion", files=files)
    return response.json()

def call_text_emotion_api(text):
    response = requests.post("https://api.ngrok.com/tunnels/hidn54domwsoa15678/text_emotion", json={"text": text})
    return response.json()

def call_fusion_api(video_result, audio_result, text_result):
    data = {
        "video": video_result,
        "audio": audio_result,
        "text": text_result
    }
    response = requests.post("https://api.ngrok.com/tunnels/hidn54domwsoa15678/fuse", json=data)
    return response.json()

# --- Streamlit UI ---
def main():
    st.title("🎬 Multimodal Emotion Detection via Dummy APIs")
    st.write("Upload a video, and we’ll call simulated APIs for emotion detection on video, audio, and text.")

    uploaded_file = st.file_uploader("📤 Upload video file", type=["mp4", "mkv", "mov", "avi"])

    if uploaded_file:
        with st.spinner("Processing..."):
            suffix = "." + uploaded_file.name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
                temp_video.write(uploaded_file.read())
                video_path = temp_video.name

            audio_path = extract_audio_ffmpeg(video_path)
            transcript = transcribe_audio(audio_path)

            video_result = call_video_emotion_api(video_path)
            audio_result = call_audio_emotion_api(audio_path)
            text_result = call_text_emotion_api(transcript)

            final_result = call_fusion_api(video_result, audio_result, text_result)

            os.remove(video_path)
            os.remove(audio_path)

        st.success("✅ Done!")
        st.subheader("📝 Transcription")
        st.write(transcript)

        st.subheader("📹 Video Emotion")
        st.write(video_result)

        st.subheader("🔊 Audio Emotion")
        st.write(audio_result)

        st.subheader("📄 Text Emotion")
        st.write(text_result)

        st.subheader("🎯 Final Emotion")
        st.markdown(f"*Emotion:* {final_result['final_emotion']}")
        st.markdown(f"*Cause:* {final_result['final_cause']}")

if _name_ == "_main_":
    main()