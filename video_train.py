import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Embedding, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
DATA_PATH = '.'  # Assuming train1.csv is in the current directory
VIDEO_PATH = 'output_scenes'  # Assuming video scenes are in this subdirectory
EMBEDDING_DIM = 100
LSTM_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 10
MAX_SEQUENCE_LENGTH = 50  # Adjust based on your statement lengths

# --- 1. Load and Preprocess Data ---
try:
    df = pd.read_csv(os.path.join(DATA_PATH, 'train1.csv'))
except FileNotFoundError:
    print(f"Error: train1.csv not found in {DATA_PATH}")
    exit()

# Extract relevant information
df['episode'] = df['episode_info'].str.split(':').str[0]
df['scene'] = df['scene_id'].astype(str)
df['video_filename'] = df['episode'].str.replace('S', '').str.replace('E', 'x') + '_scene' + df['scene'] + '.mkv'

# --- 2. Prepare Video Data ---
def extract_frames(video_path, target_size=(64, 64)):
    """Extracts a fixed number of frames from a video and resizes them."""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0 or frame_rate == 0:
            return []

        # Extract a maximum of MAX_FRAMES, evenly spaced
        MAX_FRAMES = 15  # You can adjust this
        frame_indices = np.linspace(0, total_frames - 1, min(total_frames, MAX_FRAMES), dtype=int)
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, target_size)
                frame = frame / 255.0  # Normalize pixel values
                frames.append(frame)
            else:
                break
        cap.release()
        return np.array(frames)
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return []

video_features = {}
for index, row in df.iterrows():
    video_file = os.path.join(VIDEO_PATH, row['video_filename'])
    if os.path.exists(video_file):
        frames = extract_frames(video_file)
        if frames.size > 0:
            video_features[row['video_filename']] = frames
    else:
        print(f"Warning: Video file not found: {video_file}")

# Prepare video data for training
video_input_list = []
for _, row in df.iterrows():
    if row['video_filename'] in video_features:
        video_input_list.append(video_features[row['video_filename']])
    else:
        video_input_list.append(np.zeros((1, 64, 64, 3))) # Placeholder if video not found

# Pad sequences of frames (important for RNN/LSTM)
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_video_features = pad_sequences(video_input_list, padding='post', dtype='float32')

# --- 3. Prepare Text Data (Statements and Causes) ---
statements = df['statement'].fillna('').tolist()

# Create a vocabulary and tokenize statements
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<unk>") # Adjust num_words
tokenizer.fit_on_texts(statements)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
text_sequences = tokenizer.texts_to_sequences(statements)
padded_text_sequences = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Prepare emotion labels
labels = df['emotion'].tolist()
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
from tensorflow.keras.utils import to_categorical
categorical_labels = to_categorical(encoded_labels, num_classes=num_classes)

# --- 4. Define the Model Architecture ---
# Video Feature Extractor (Simplified CNN)
video_input = Input(shape=(None, 64, 64, 3))  # TimeDistributed input for frames
video_conv = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(video_input)
video_pool = TimeDistributed(MaxPooling2D((2, 2)))(video_conv)
video_conv2 = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(video_pool)
video_pool2 = TimeDistributed(MaxPooling2D((2, 2)))(video_conv2)
video_flatten = TimeDistributed(Flatten())(video_pool2)
video_lstm = LSTM(LSTM_UNITS)(video_flatten)

# Text Feature Extractor (Embedding and LSTM)
text_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
text_embedding = Embedding(vocab_size, EMBEDDING_DIM)(text_input)
text_lstm = LSTM(LSTM_UNITS)(text_embedding)

# Combined Features for Emotion Prediction
merged = Concatenate()([video_lstm, text_lstm])
emotion_output = Dense(num_classes, activation='softmax', name='emotion_output')(merged)

# Cause Prediction (Simplified - requires more sophisticated approach)
# This is a placeholder. Generating free-form text as a "cause" is much harder.
# A more realistic approach would involve identifying relevant parts of the conversation
# or training a separate model for cause extraction/generation.
cause_output = Dense(MAX_SEQUENCE_LENGTH, activation='sigmoid', name='cause_output')(text_input) # Just passing text through for now - not meaningful

# Define the model with multiple outputs
model = Model(inputs=[video_input, text_input], outputs=[emotion_output, cause_output])

# Compile the model
model.compile(optimizer='adam',
              loss={'emotion_output': 'categorical_crossentropy', 'cause_output': 'binary_crossentropy'}, # Adjust loss for cause
              metrics={'emotion_output': 'accuracy'})

# --- 5. Train the Model ---
# Split data into training and validation sets
video_train, video_val, text_train, text_val, emotion_train, emotion_val = train_test_split(
    padded_video_features, padded_text_sequences, categorical_labels, test_size=0.2, random_state=42
)

# Train the model
history = model.fit(
    {'video_input': video_train, 'text_input': text_train},
    {'emotion_output': emotion_train, 'cause_output': np.zeros_like(text_train)}, # Placeholder for cause target
    validation_data=({'video_input': video_val, 'text_input': text_val},
                     {'emotion_output': emotion_val, 'cause_output': np.zeros_like(text_val)}), # Placeholder
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# --- 6. Evaluation (Optional) ---
loss, emotion_loss, cause_loss, emotion_accuracy = model.evaluate(
    {'video_input': video_val, 'text_input': text_val},
    {'emotion_output': emotion_val, 'cause_output': np.zeros_like(text_val)},
    verbose=0
)
print(f"Validation Emotion Accuracy: {emotion_accuracy * 100:.2f}%")

# --- 7. Prediction Function ---
def predict_emotion_and_cause(video_file, statement):
    """
    Predicts the emotion and (conceptual) cause for a given video file and statement.
    """
    frames = extract_frames(video_file)
    if not frames.size > 0:
        print(f"Error: Could not process video {video_file} for prediction.")
        return None, None

    padded_video = pad_sequences([frames], padding='post', maxlen=padded_video_features.shape[1], dtype='float32')

    text_sequence = tokenizer.texts_to_sequences([statement])
    padded_statement = pad_sequences(text_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    predictions = model.predict([padded_video, padded_statement])
    emotion_prediction = predictions[0]
    cause_prediction = predictions[1]

    predicted_emotion_index = np.argmax(emotion_prediction)
    predicted_emotion = label_encoder.inverse_transform([predicted_emotion_index])[0]

    # --- Interpreting "Cause" (Conceptual and Basic) ---
    # This is a very basic interpretation. A real cause would require more sophisticated methods.
    # Here, we are just using the input statement as a very rough proxy.
    predicted_cause = statement

    return predicted_emotion, predicted_cause

# --- Example Usage ---
example_video = os.path.join(VIDEO_PATH, '1x02_scene2.mkv') # Assuming this file exists
example_statement = "I can't believe she said that to him!"

if os.path.exists(example_video):
    predicted_emotion, predicted_cause = predict_emotion_and_cause(example_video, example_statement)
    if predicted_emotion:
        print(f"Predicted Emotion for {example_video}: {predicted_emotion}")
        print(f"Predicted Cause: {predicted_cause}")
else:
    print(f"Example video file not found: {example_video}")