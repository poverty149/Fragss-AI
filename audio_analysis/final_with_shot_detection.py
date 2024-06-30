import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from panns_inference import AudioTagging, labels
from panns_inference.models import Cnn14
from models import *

# Define the function to extract features and perform audio tagging
def generate_features(file_path):
    audio, sr = librosa.load(file_path)

    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    rms = librosa.feature.rms(y=audio)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

    feature_vector = np.concatenate(
        (chroma_stft, spectral_centroid, spectral_bandwidth, spectral_rolloff, rms, zero_crossing_rate),
        axis=0
    )

    return feature_vector

def extract_audio_tags(audio_path, at):
    audio, _ = librosa.load(audio_path, sr=32000)
    audio = audio[None, :]  # Add batch dimension
    clipwise_output, _ = at.inference(audio)
    labels_df = pd.read_csv('/Users/parvathyuk/panns_data/class_labels_indices.csv')
    labels_df['prediction'] = clipwise_output.reshape(-1)
    top_tags = labels_df.sort_values(by='prediction', ascending=False).head(5)
    significant_tags = labels_df[labels_df['prediction'] > 0.5]
    return top_tags, significant_tags

# Initialize the AudioTagging model
at = AudioTagging(
    checkpoint_path="/Users/parvathyuk/panns_data/Cnn10_mAP=0.380.pth",
    model=Cnn10(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
)

def calculate_histogram(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
    return cv2.normalize(hsv_hist, hsv_hist).flatten()

def preprocess_frame(frame, size):
    return cv2.resize(frame, size)

def analyze_initial_frames(video_path, sample_size=50, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return None
    
    good_matches_counts = []
    ret, prev_frame = cap.read()
    if not ret:
        logging.error("Error reading the first frame.")
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)
    
    frame_count = 0

    while frame_count < sample_size:
        frame_count += 1
        for _ in range(frame_skip):
            ret, frame = cap.read()
            if not ret:
                break
        
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(prev_des, des)
        
        good_matches = [m for m in matches if m.distance < 50]
        good_matches_counts.append(len(good_matches))
        
        prev_gray = gray
        prev_kp, prev_des = kp, des

    cap.release()
    logging.info(f"Initial analysis complete. Analyzed {len(good_matches_counts)} frames.")
    return good_matches_counts

def calculate_dynamic_threshold(good_matches_counts, percentile=10):
    threshold = np.percentile(good_matches_counts, percentile)
    logging.info(f"Dynamic match threshold set to {threshold:.2f} based on the {percentile}th percentile.")
    return threshold

def process_frames(video_path, start_frame, end_frame, step_size, clip_limit=2, method='sift'):
    good_matches_count = analyze_initial_frames(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if method == 'hist':
        shot_boundaries = []
        prev_bound = start_frame
        prev_hist = None
        frames = []
        frame_number = start_frame

        while frame_number < end_frame:
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Error reading frame: {frame_number}")
                break

            if frame_number % step_size == 0:
                frame = preprocess_frame(frame, (224, 224))
                frames.append((frame_number, frame))

            hist = calculate_histogram(frame)

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if diff < 0.7:
                    if (frame_number - prev_bound) > min_clip_length:
                        shot_boundaries.append((prev_bound - start_frame, frame_number - start_frame))
                        prev_bound = frame_number
                if len(shot_boundaries) >= clip_limit:
                    break

            prev_hist = hist
            frame_number += 1
    else:
        shot_boundaries = []
        prev_bound = start_frame
        frames = []
        frame_number = start_frame
        sift = cv2.SIFT_create()
        prev_kp, prev_des = None, None
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        match_threshold = calculate_dynamic_threshold(good_matches_count, percentile=10)

        while frame_number < end_frame:
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Error reading frame: {frame_number}")
                break

            if frame_number % step_size == 0:
                frame = preprocess_frame(frame, (224, 224))
                frames.append((frame_number, frame))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            if prev_des is not None:
                matches = bf.match(prev_des, des)
                good_matches = [m for m in matches if m.distance < 50]
        
                logging.info(f"Frame {frame_number}: {len(good_matches)} good matches found.")
                
                if len(good_matches) < match_threshold:
                    logging.info(f"Shot boundary detected at frame {frame_number}.")
                    shot_boundaries.append((prev_bound - start_frame, frame_number - start_frame))
                    prev_bound = frame_number
                prev_kp, prev_des = kp, des
                if len(shot_boundaries) >= clip_limit:
                    break

            prev_hist = hist
            frame_number += 1

    cap.release()
    logging.info(f"Shot boundary detection complete. {len(shot_boundaries)} boundaries found.")

    return frames, shot_boundaries

def parallel_frame_processing(video_path, num_threads=4, clip_limit=4):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    step_size = fps // 10
    cap.release()
    
    chunk_size = total_frames // num_threads
    futures = []

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start_frame = i * chunk_size
            end_frame = (i + 1) * chunk_size if i != num_threads - 1 else total_frames
            futures.append(executor.submit(process_frames, video_path, start_frame, end_frame, step_size, clip_limit))
        
        frames = []
        shot_boundaries = []
        len_frames = 0

        for future in as_completed(futures):
            result_frames, result_boundaries = future.result()
            frames.extend(result_frames)

            adjusted_boundaries = [(boundary[0] + len_frames, boundary[1] + len_frames) for boundary in result_boundaries]
            len_frames += len(result_frames) * step_size
            shot_boundaries.extend(adjusted_boundaries)

    end_time = time.time()
    logging.info(f"Processing time: {end_time - start_time} seconds")
    
    return frames, shot_boundaries

def save_clips_with_audio_tags(frames, shot_boundaries, video_path, output_dir="clips"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, (start_frame, end_frame) in enumerate(shot_boundaries):
        out = cv2.VideoWriter(f"{output_dir}/clip_{i}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
        
        for frame_number, frame in frames:
            if start_frame <= frame_number <= end_frame:
                original_frame = preprocess_frame(frame, (width, height))
                out.write(original_frame)

        out.release()

        # Extract the corresponding audio segment
        start_time = start_frame / fps
        end_time = end_frame / fps
        audio_clip_path = f"{output_dir}/clip_{i}.wav"
        librosa.output.write_wav(audio_clip_path, audio[int(start_time*sr):int(end_time*sr)], sr)
        
        # Perform audio tagging
        top_tags, significant_tags = extract_audio_tags(audio_clip_path, at)
        
        # Save tags to a text file
        tags_path = f"{output_dir}/clip_{i}_tags.txt"
        with open(tags_path, 'w') as f:
            f.write("Top 5 Audio Tags:\n")
            for _, row in top_tags.iterrows():
                f.write(f"{row['label']}: {row['prediction']:.3f}\n")
            f.write("\nSignificant Audio Tags (threshold > 0.5):\n")
            for _, row in significant_tags.iterrows():
                f.write(f"{row['label']}: {row['prediction']:.3f}\n")

# Example usage
video_path = 'sample.mp4'
frames, shot_boundaries = parallel_frame_processing(video_path, clip_limit=4)
save_clips_with_audio_tags(frames, shot_boundaries, video_path)
logging.info(f"Total frames processed: {len(frames)}")
logging.info(f"Shot boundaries detected: {shot_boundaries}")
