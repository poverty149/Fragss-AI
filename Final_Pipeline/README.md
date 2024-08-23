## Viral Clip Generation of Video Game Streams

### Introduction

This folder contains all the relevant code required to process long videos and generate viral clips.

### Instructions

The code used for training the video and audio classification models can be found in video_classification_training.py and audio_classification_training.py respectively.

For the audio tagging section you have to download the pretrained weights from this [link](https://zenodo.org/records/3987831)

Download the pretrained weights for the action recognition model used while training from this [link](https://drive.google.com/file/d/1rlZ-xTkTMjgWKiQFUedRnHlDgQwx6yTm/view)

You will have to generate the scaler.pkl file, video_virality_model.h5 and model_sound.weights.h5 using audio_classification_training.py and video_classification_training.py. These processes will take time but it only has to be implemented once during training. Additionally you may need to adjust the paths in the code, but this is fairly self explanatory.

### Dataset
I have provided a copy of the csv file containing the metadata of the viral and non-viral videos used for training. You can alternatively collect your own videos by using the tiktok scraper and youtube downloader files located in the virality folder.
