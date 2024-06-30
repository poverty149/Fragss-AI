import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import pandas as pd
from models import *


def generate_features(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path)

    # Extract features
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    rms = librosa.feature.rms(y=audio)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    # Concatenate the features into a single feature vector
    feature_vector = np.concatenate(
        (chroma_stft, spectral_centroid, spectral_bandwidth, spectral_rolloff, rms, zero_crossing_rate),
        axis=0
    )

    return feature_vector

if __name__=='__main__':

    file_path='sample.mp4'
    audio, sr=librosa.load(file_path)
    at=AudioTagging(checkpoint_path="/Users/parvathyuk/panns_data/Cnn10_mAP=0.380.pth",model=Cnn10(sample_rate=32000, window_size=1024,hop_size=320,
                                                                    mel_bins=64, fmin=50, fmax=14000,classes_num=527))
    sound = audio[None, :]
    (clipwise_output, embedding) = at.inference(sound)
    labels = pd.read_csv('/Users/parvathyuk/panns_data/class_labels_indices.csv')
    labels['prediction'] = clipwise_output.reshape(-1)
    labels.sort_values(by=['prediction'],ascending=False)