# Fragss-AI
# Video Segmentation Tool

The Video Segmentation Tool is a Python-based application that allows users to segment long video files into smaller clips based on various criteria, such as shot boundaries, object detection, and audio event detection.

## Features

- **Shot Boundary Detection**: Identify transitions between shots or scenes in the input video using histogram-based comparison techniques.
- **Video Clip Generation**: Extract video segments based on the detected shot boundaries and save them as individual clips.
- **Audio Clipping (Optional)**: Clip the audio of the input video based on the detected shot boundaries and save the audio segments separately.
- **Video and Audio Combination**: Combine the generated video and audio clips into the final output files.
- **GUI-based Interface**: Provide a user-friendly graphical interface for selecting the input video, output directory, and configuring the segmentation options.
- **Filtering Options**: Allow users to enable or disable object detection and audio event detection as additional filtering criteria for the video segmentation process.

## Prerequisites

- Python 3.6 or newer
- The following Python libraries:
  - `opencv-python`
  - `tkinter`
  - `moviepy`
  - `numpy`

## Usage
1. Instal the required libraries and run the program.
2. In the GUI, click the "Select Video" button to choose the input video file.
3. Click the "Select Output" button to choose the output directory where the segmented clips will be saved.
4. (Optional) Check the "Object Detection" and/or "Audio Event Detection" checkboxes to enable those filtering criteria.
5. Click the "Segment Video" button to start the video segmentation process.
6. The segmented video and audio clips will be saved in the selected output directory.

## Things to work on
1. Might need to adjust the threshold value for feature detection further.
2. Correct glitches occurring after saving the clips.
3. Need to implement faster processing techniques, currently the extraction step takes too much time.
4. Collecting an action recognition dataset that would be appropriate for video games alone (For the meantime, stick to UCF101 and kinetics).
   

## Contributing

Contributions to the Video Segmentation Tool are welcome! If you find any issues or have ideas for improvements, please feel free to submit a pull request or open an issue on the [GitHub repository]

## Acknowledgments

- The `shot_boundary_detection`, `clip_video`, `clip_audio`, and `combine_video_audio` functions were adapted from the `Video_Segmentation_Tool.py` file.
- The GUI implementation was based on the Tkinter library.
