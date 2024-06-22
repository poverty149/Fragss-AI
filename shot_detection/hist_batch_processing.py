
import cv2
import numpy as np
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_histogram(frame):
    """
    Convert the frame to HSV space and then calculate its histogram.
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
    return cv2.normalize(hsv_hist, hsv_hist).flatten()

def preprocess_frame(frame, size):
    """
    Resize the frame to the given size.
    """
    return cv2.resize(frame, size)

def process_frames(video_path, start_frame, end_frame, step_size, clip_limit=2,method='sift'):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if method=='hist':

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
                    if (frame_number-prev_bound)>min_clip_length:
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
        match_threshold=500
    

        while frame_number < end_frame:
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Error reading frame: {frame_number}")
                break

            if frame_number % step_size == 0:
                frame = preprocess_frame(frame, (224, 224))
                frames.append((frame_number, frame))

            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    step_size=fps//10
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

            # Adjust shot boundaries to maintain correct values relative to the entire video
            adjusted_boundaries = [(boundary[0] + len_frames, boundary[1] + len_frames) for boundary in result_boundaries]
            len_frames += len(result_frames) * step_size
            print(len_frames)
            shot_boundaries.extend(adjusted_boundaries)

    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")
    
    return frames, shot_boundaries

def save_clips_using_processed_frames(frames, shot_boundaries, video_path, output_dir="clips"):
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
                # Convert frame back to original size
                original_frame = preprocess_frame(frame, (width, height))
                out.write(original_frame)

        out.release()


# Example usage
video_path = 'sample.mp4'
step_size = 60
clip_limit = 4
min_clip_length = 2
frames, shot_boundaries = parallel_frame_processing(video_path, clip_limit=clip_limit)

# Save the detected clips into separate MP4 files using processed frames
save_clips_using_processed_frames(frames, shot_boundaries, video_path)

# frames now contain the processed frames along with their frame numbers
logging.info(f"Total frames processed:{len(frames)}")
logging.info(f"Shot boundaries detected: {shot_boundaries}")
