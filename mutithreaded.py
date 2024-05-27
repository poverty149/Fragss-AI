import cv2
import numpy as np
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def calculate_histogram(frame):
    """
    Convert the frame to HSV space and then calculate its histogram.
    """
    if frame.dtype != np.uint8:
        frame = cv2.convertScaleAbs(frame)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
    return hsv_hist

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
    frame_width,frame_height=img.shape[0],img.shape[1]
    
    return processed_frames,frame_width,frame_height

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

def main(input_dir, output_dir, video_path, step_size=1):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    prev_hist = None
    scene_changes = []
    min_frames_between_changes = int(3 * fps)  # Minimum frames corresponding to 3 seconds
    processed_frames,frame_width,frame_height = read_frames(input_dir, step_size=step_size)
    frame_number = 0

    for frame in tqdm(processed_frames, desc="Processing frames"):
        cur_hist = calculate_histogram(frame)

        if prev_hist is not None:
            hist_diff = cv2.compareHist(prev_hist, cur_hist, cv2.HISTCMP_CORREL)

            if hist_diff < 0.7:  # Adjust this threshold based on your requirements
                if not scene_changes or (frame_number - scene_changes[-1] >= min_frames_between_changes):
                    scene_changes.append(frame_number)
                    
        prev_hist = cur_hist
        frame_number += step_size

    scene_changes.append(len(processed_frames))  # Ensure the last segment is processed

    start_frame = 0
    cur_scene = 0
    frame_size = (frame_width, frame_height)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for change_frame in scene_changes:
            output_path = os.path.join(output_dir, f'scene_{cur_scene}.mp4')
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
        
        # Wait for all tasks to complete
        for future in tqdm(futures, desc="Saving scenes"):
            future.result()

if __name__ == '__main__':
    input_dir = 'extracted'  # Directory where preprocessed frames are stored
    output_dir = 'output_scenes'  # Directory to save the scene segments
    video_path = 'new_seg.mp4'  # Path to the input video file
    os.makedirs(output_dir, exist_ok=True)
    main(input_dir, output_dir, video_path)
