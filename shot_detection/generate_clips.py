import cv2
import numpy as np
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import moviepy.editor as mp

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

def analyze_initial_frames(video_path, sample_size=50, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return None
    
    # Read initial frames for analysis
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
        for _ in range(frame_skip):  # Skip frames
            ret, frame = cap.read()
            if not ret:
                break
        
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        
        # Match features between frames
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(prev_des, des)
        
        # Filter good matches
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

def process_frames(video_path, start_frame, end_frame, step_size,min_frames=50,method='hist'):
    if method=='sift':
        good_matches_count=analyze_initial_frames(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    original_frames=[]
    

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
                original_frames.append((frame_number, frame))
                frame = preprocess_frame(frame, (224, 224))
                frames.append((frame_number, frame))

            hist = calculate_histogram(frame)

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if diff < 0.7:
                    if (frame_number-prev_bound)>min_frames:
                        shot_boundaries.append((prev_bound - start_frame, frame_number - start_frame))
                        prev_bound = frame_number

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
        match_threshold=calculate_dynamic_threshold(good_matches_count,percentile=10)
    

        while frame_number < end_frame:
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Error reading frame: {frame_number}")
                break

            if frame_number % step_size == 0:
                original_frames.append((frame_number, frame))
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
            
            prev_hist = hist
            frame_number += 1

    cap.release()
    logging.info(f"Shot boundary detection complete. {len(shot_boundaries)} boundaries found.")

    return original_frames, shot_boundaries

def parallel_frame_processing(video_path, num_threads=4,min_clip_length=2):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    step_size=fps//10
    print(step_size)
    cap.release()
    min_frames=fps*min_clip_length
    chunk_size = total_frames // num_threads
    futures = []

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start_frame = i * chunk_size
            end_frame = (i + 1) * chunk_size if i != num_threads - 1 else total_frames
            futures.append(executor.submit(process_frames, video_path, start_frame, end_frame, step_size,min_frames))
        
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
    step_size=fps//10
    final_fps=np.floor(fps/step_size)
    print(final_fps)
    
    cap.release()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, (start_frame, end_frame) in enumerate(shot_boundaries):
        
        out = cv2.VideoWriter(f"{output_dir}/clip_{i}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), final_fps, (width, height))
        for frame_number, frame in frames:
            if start_frame <= frame_number <= end_frame:
                # Convert frame back to original size
                # original_frame = preprocess_frame(frame, (width, height))
                out.write(frame)

        out.release()
    
        

def save_clips_using_processed_frames_with_audio(frames, shot_boundaries, video_path, output_dir="clips"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step_size=fps//10
    final_fps=np.floor(fps/step_size)
    print(final_fps)
    
    cap.release()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, (start_frame, end_frame) in enumerate(shot_boundaries):
        
        out = cv2.VideoWriter(f"{output_dir}/clip_{i}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), final_fps, (width, height))
        video = mp.VideoFileClip(video_path).subclip(start_frame / fps, end_frame / fps)
        audio = video.audio
        audio.write_audiofile(f"{output_dir}/clip_{i}.wav")
        for frame_number, frame in frames:
            if start_frame <= frame_number <= end_frame:
                # Convert frame back to original size
                # original_frame = preprocess_frame(frame, (width, height))
                out.write(frame)

        out.release()
        final_clip = mp.VideoFileClip(f"{output_dir}/clip_{i}.mp4").set_audio(audio)
        final_clip.write_videofile(f"{output_dir}/final_clip_{i}.mp4", codec="libx264",audio_codec='aac')
        
        # Clean up temporary files
        os.remove(f"{output_dir}/clip_{i}.mp4")
        os.remove(f"{output_dir}/clip_{i}.wav")
        
        

if __name__=='__main__':

    video_path = 'new_seg.mp4'
    min_clip_length = 2
    st = time.time()
    frames, shot_boundaries = parallel_frame_processing(video_path)
    et = time.time()
    logging.info(f"Total processing time: {et - st} seconds")

    # Save clips with audio
    save_clips_using_processed_frames_with_audio(frames, shot_boundaries, video_path)
