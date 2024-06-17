import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Load metadata
csv_file = 'valorant2.csv'
df = pd.read_csv(csv_file)
video_ids = df['description'].values  # Ensure this column corresponds to video file names

# Path to the directory containing videos
video_dir = './training_videos'
output_file = 'video_features.npy'

# Load pre-trained ResNet50 model + higher level layers
base_model = ResNet50(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Function to extract features from a single frame
def extract_frame_features(frame):
    try:
        img = cv2.resize(frame, (224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = model.predict(img)
        return features.flatten()
    except Exception as e:
        print(f"Error processing frame: {e}")
        return np.zeros((2048,))

# Function to process a video and extract features
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_features = extract_frame_features(frame)
        features.append(frame_features)
    cap.release()
    if features:
        features = np.array(features)
        video_features = np.mean(features, axis=0)  # Aggregate frame features
    else:
        video_features = np.zeros((2048,))
    return video_features

# Process all videos and save the features
video_features = []
for video_id in video_ids:
    video_file = f"{video_id[:-1]}.mp4"  # Ensure video files are named according to video_id
    video_path = os.path.join(video_dir, video_file)
    if os.path.exists(video_path):
        print(f"Processing {video_file}...")
        video_feature = process_video(video_path)
        print(f"Extracted features shape for {video_file}: {video_feature.shape}")  # Debug print
        video_features.append(video_feature)
    else:
        print(f"Video file {video_file} not found. Skipping.")
        video_features.append(np.zeros((100352,)))  # Assuming ResNet50 output size

print(f"All video features collected, shapes: {[vf.shape for vf in video_features]}")  # Debug print

video_features = np.array(video_features)
np.save(output_file, video_features)
print(f"Saved video features to {output_file}")
