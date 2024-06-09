import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video at the specified frame rate.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    return extracted_count

def calculate_histogram(frame):
    """
    Convert the frame to HSV space and then calculate its histogram.
    """
    if frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
    return hsv_hist

def extract_features(frame, model):
    """
    Extract features from a frame using a pre-trained model.
    """
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    features = model.predict(frame)
    return features.flatten()

def read_frames(input_dir, normalize=True, step_size=1):
    """
    Reads the frames from the preprocessed directory and normalizes the images for further processing.
    Adjust the step size value according to your needs or implement a computation factor for this parameter later.
    """
    processed_frames = []
    cur_frame_number = 0

    for frame_path in sorted(glob.glob(input_dir + "/*.jpg")):
        if cur_frame_number % step_size != 0:
            cur_frame_number += 1
            continue
        img = cv2.imread(frame_path)
        if normalize:
            img = img / 255.0
        processed_frames.append(img)
        cur_frame_number += 1
    frame_width, frame_height = img.shape[1], img.shape[0]
    
    return processed_frames, frame_width, frame_height

def detect_shot_boundaries(frames, features, histograms, threshold=0.07):
    """
    Detect shot boundaries based on feature differences and histogram differences.
    """
    shot_boundaries = []
    prev_feature = features[0]
    prev_hist = histograms[0]

    for i in range(1, len(features)):
        feature_diff = np.linalg.norm(features[i] - prev_feature)
        hist_diff = cv2.compareHist(prev_hist, histograms[i], cv2.HISTCMP_CORREL)
        
        if feature_diff > threshold or hist_diff < 0.7:  # Combined condition
            shot_boundaries.append(i)
        
        prev_feature = features[i]
        prev_hist = histograms[i]

    return shot_boundaries

def process_frame_range(start_idx, end_idx, processed_frames, output_path, fps, frame_size):
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        frame_size
    )
    for frame_idx in range(start_idx, end_idx):
        if frame_idx < len(processed_frames):
            out.write((processed_frames[frame_idx] * 255).astype(np.uint8))  # Convert back to uint8 for writing
    out.release()

def main(input_dir, output_dir, video_path, step_size=1, threshold=0.7):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print("Reading frames and extracting features...")
    processed_frames, frame_width, frame_height = read_frames(input_dir, step_size=step_size)
    base_model = VGG16(weights='imagenet', include_top=False) # Also try ResNet models
    model = Model(inputs=base_model.input, outputs=base_model.output)
    
    features = []
    histograms = []
    for frame in tqdm(processed_frames, desc="Extracting features"):
        features.append(extract_features(frame, model))
        histograms.append(calculate_histogram(frame))
    
    print("Detecting shot boundaries...")
    shot_boundaries = detect_shot_boundaries(processed_frames, features, histograms, threshold)
    
    print("Saving shots...")
    start_frame = 0
    cur_scene = 0
    frame_size = (frame_width, frame_height)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for change_frame in shot_boundaries:
            output_path = os.path.join(output_dir, f'scene_{cur_scene:04d}.mp4')
            futures.append(executor.submit(
                process_frame_range,
                start_frame,
                change_frame,
                processed_frames,
                output_path,
                fps,
                frame_size
            ))
            cur_scene += 1
            start_frame = change_frame
        
        # Ensure the last segment is processed
        output_path = os.path.join(output_dir, f'scene_{cur_scene:04d}.mp4')
        futures.append(executor.submit(
            process_frame_range,
            start_frame,
            len(processed_frames),
            processed_frames,
            output_path,
            fps,
            frame_size
        ))
        
        # Wait for all tasks to complete
        for future in tqdm(futures, desc="Saving scenes"):
            future.result()

if __name__ == "__main__":
    input_dir = 'extracted'  # Directory where preprocessed frames are stored
    output_dir = 'output_scenes'  # Directory to save the scene segments
    video_path = 'sample.mp4'  # Path to the input video file
    os.makedirs(output_dir, exist_ok=True)
    main(input_dir, output_dir, video_path)
