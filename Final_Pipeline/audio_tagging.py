import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




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
relevant_tags=["Gunshot, gunfire","Machine gun","Cheering","Artillery Fire","Explosion","Applause","Clapping","Shout","Screaming","Laughter","Siren","Alarm","Fusillade","Smash, crash","Whack, thwack","Shatter","Music","Speech"]

def extract_audio_tags(audio_path, at):
    audio, _ = librosa.load(audio_path, sr=32000)
    audio = audio[None, :]  # Add batch dimension
    clipwise_output, _ = at.inference(audio)
    
    # Load class labels
    labels_df = pd.read_csv('/Users/parvathyuk/panns_data/class_labels_indices.csv')
    labels_df['prediction'] = clipwise_output.reshape(-1)
    
    # Filter relevant tags
    relevant_df = labels_df[labels_df['display_name'].isin(relevant_tags)]
    relevant_tags_with_predictions = relevant_df.sort_values(by='prediction', ascending=False)

    return relevant_tags_with_predictions

def extract_audio_tags_top(audio_path, at):
    audio, _ = librosa.load(audio_path, sr=32000)
    audio = audio[None, :]  # Add batch dimension
    clipwise_output, _ = at.inference(audio)
    labels_df = pd.read_csv('/Users/parvathyuk/panns_data/class_labels_indices.csv')
    labels_df['prediction'] = clipwise_output.reshape(-1)
    top_tags = labels_df.sort_values(by='prediction', ascending=False).head(5)
    significant_tags = labels_df[labels_df['prediction'] > 0.5]
    return top_tags, significant_tags
def audio_tag_score(audio_tags):
    relevant_tags = [
        "Gunshot, gunfire", "Machine gun", "Cheering", "Artillery Fire",
        "Explosion", "Applause", "Clapping", "Shout", "Screaming", "Laughter",
        "Siren", "Alarm", "Fusillade", "Smash, crash", "Whack, thwack", "Shatter",
        "Music", "Speech"
    ]

    # Thresholds
    special_threshold = 0.15
    general_threshold = 0.05

    # Initialize scores
    speech_music_score = 0
    other_tags_score = 0

    for index, row in audio_tags.iterrows():
        tag = row['display_name']
        prediction = row['prediction']
        
        if tag in ["Speech", "Music"]:
            if prediction >= special_threshold:
                speech_music_score += prediction * 10  # Each of Speech and Music can contribute up to 5 points each
        elif tag in relevant_tags:
            if prediction >= general_threshold:
                other_tags_score += (prediction-general_threshold)/general_threshold * 15 / len(relevant_tags)  # Distribute the 15 points among other relevant tags

    # Ensure the scores are capped at their respective maximum values
    speech_music_score = min(speech_music_score, 10)
    other_tags_score = min(other_tags_score, 15)

    # Total score is the sum of both parts
    total_score = speech_music_score + other_tags_score

    return total_score
