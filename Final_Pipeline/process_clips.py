import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
from panns_inference import AudioTagging
from panns_inference.models import Cnn14
from audio_tagging import *
from video_classification import *
from audio_classification import *


# Function to preprocess input for MobileNetV2
def preprocess_input(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame)
    return frame

def calculate_audio_energy(audio_path):
    audio, sr = librosa.load(audio_path, sr=32000)
    energy = np.sum(np.square(audio))
    return energy

def assign_points(audio_energy):
    if audio_energy < 200:
        return 2
    elif audio_energy < 500:
        return 4
    elif audio_energy < 1000:
        return 6
    elif audio_energy < 2000:
        return 8
    else:
        return 10
# Function to extract audio tags

# Process clips to calculate scores
def process_clips(clips_dir, feature_extraction_model, classification_model, at):
    clip_scores = []
    for clip_name in os.listdir(clips_dir):
        if clip_name.endswith(('.mp4', '.avi', '.mov')):
            
            # clip_name='final_clip_3.mp4'
            clip_path = os.path.join(clips_dir, clip_name)
            # Extract audio path (assuming audio is extracted from video)
            audio_path = clip_path.replace('.mp4', '.wav').replace('.avi', '.wav').replace('.mov', '.wav')
            audio_path = audio_path.replace('final_','')
            # Extract audio tags
            audio_tags = extract_audio_tags(audio_path, at)
            
            audio_prediction = predict_viral(clip_path)
            audio_classification_score = audio_prediction[0][1] * 30
            
            audio_energy = calculate_audio_energy(audio_path)
            # print(audio_energy)
            
            # Normalize audio energy
            final_audio_energy = assign_points(audio_energy)
            
            # Calculate audio score
            audio_tag_score = sum(audio_tags['prediction'][:5])*20  # Replace with your logic to calculate audio score from tags
            # print(audio_classification_score,'audio')
            # print(audio_tag_score,'tag',clip_name)
            # Predict video virality
            video_prediction = predict_virality(clip_path, feature_extraction_model, classification_model)
            print(video_prediction)
            video_score = video_prediction[0][0] * 30
            # print(video_score,'vid')
            
            # Calculate combined score
            combined_score = video_score+audio_classification_score +audio_tag_score+final_audio_energy 
            print(combined_score,'com')
            clip_scores.append((clip_name, combined_score))

    
    # Sort clips by combined score
    clip_scores.sort(key=lambda x: x[1], reverse=True)
    return clip_scores

def process_clips_from_folder(clips_dir='./static/clips'):
    # Constants
    IMAGE_SIZE = (224, 224)  # Size to resize frames

    # Load the models
    feature_extraction_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    classification_model = load_model('video_virality_model.h5')
    classification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the model
    # Initialize the AudioTagging model
    at = AudioTagging(
        checkpoint_path="/Users/parvathyuk/panns_data/Cnn14_mAP=0.431.pth",
        model=Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    )
    # Example usage
    top_n = 5
    clip_scores = process_clips(clips_dir, feature_extraction_model, classification_model, at)
    top_clips = clip_scores[:top_n]

    for clip_name, score in top_clips:
        print(f"Clip: {clip_name}, Combined Score: {score}")
    return top_clips
