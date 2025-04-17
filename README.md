# Emotion Cause Extraction from Videos using Audio, Text, and Visual Cues 🎥🔊📝

This project identifies emotions and their causes in video clips (e.g., Friends TV show) using multimodal learning: **text (transcripts), audio, and video**. The project pipeline involves dataset preparation, annotation, model training, and a Streamlit frontend for demo.

## 🗂️ File Overview

| File Name                                     | Description |
|----------------------------------------------|-------------|
| `Roberta_Text_model.ipynb`                   | Text-based emotion extraction using RoBERTa |
| `audio_model.ipynb`                          | Audio-based emotion classification model |
| `code_compression.py`                        | Compress video files for faster processing |
| `dev.csv`, `test.csv`, `train.csv`           | Prepared datasets (split into dev/test/train) |
| `emotioncause_group2.pptx`                   | Project presentation |
| `extract_audio.py`                           | Extract audio from video files |
| `preparing_data.py`                          | Annotate and pair emotion-cause data |
| `res_paper_friends.pdf`                      | Reference paper used for understanding the task |
| `video_aufio_text_emotion_cause_extractions.py` | Main Streamlit app for demo |
| `video_trimming.py`                          | Trim videos based on timestamps (from annotation) |

---

## 🚀 How to Run the Project

### 1️⃣ Dataset Preparation

1. **Download the raw video dataset**:
   - [📥 Download Friends Dataset](https://drive.google.com/drive/folders/1JqyfZKdkhmgymOVxZbfePG6XqQGUGoZF?usp=drive_link)
   - Place the videos in a folder named `raw_videos`.

2. **Trim the Videos**:
   ```bash
   python video_trimming.py
   ```

3. **Compress the Trimmed Videos**:
   ```bash
   python code_compression.py
   ```

4. **Extract Audio**:
   ```bash
   python extract_audio.py
   ```

5. **Extract Text using Whisper** (Run separately using OpenAI Whisper or Whisper API).
   - Save the output transcripts in a suitable format.

---

### 2️⃣ Annotation & Data Preparation

Run the following script to properly annotate emotion-cause pairs:
```bash
python preparing_data.py
```

Ensure `train.csv`, `dev.csv`, and `test.csv` are populated accordingly.

---

### 3️⃣ Running the Streamlit App

There are 3 separate model notebooks:
- `Roberta_Text_model.ipynb` for text
- `audio_model.ipynb` for audio
- (Add your video model similarly if available)

Each should:
- Be run in separate Colab notebooks
- Expose an **API endpoint using ngrok**

💡 Make sure you get your **ngrok auth token** and run the following in each notebook:
```bash
!ngrok authtoken <your-ngrok-auth-token>
```

Get the **3 API links** (text, audio, video) from ngrok and **paste them inside** `video_aufio_text_emotion_cause_extractions.py`.

---

### 4️⃣ Running the Frontend

Install Streamlit:
```bash
pip install streamlit
```

Run the Streamlit app:
```bash
streamlit run video_aufio_text_emotion_cause_extractions.py
```

🎉 You can now upload a video and the app will display:
- Detected emotions
- Their possible causes using multimodal inference

---

## 📌 Notes

- Make sure the formats for text and audio data match with what's expected in each model.
- Refer to the `.csv` datasets for training and validation purposes.
- You may modify the `video_aufio_text_emotion_cause_extractions.py` to adjust UI or API integration.

---

## 📚 Reference

- `res_paper_friends.pdf`: Contains the core academic idea this project is based on.
- `emotioncause_group2.pptx`: Team presentation covering approach, models used, and evaluation.

---

## ✨ Future Improvements

- Unified multimodal model instead of separate APIs
- Auto Whisper integration
- Model deployment using Docker or HuggingFace Spaces

---

## 👨‍💻 Contributors

2101CS03 - Achhada Hiren Rajkumar

2101CS07 - Ali Haider

2101CS79 - Vansh Singh
