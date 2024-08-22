import os
import subprocess
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

#Define model
def build_classification_model(input_shape):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(0.01)))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def extract_audio_from_videos(video_dir, audio_dir):
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    
    for filename in os.listdir(video_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_dir, filename)
            audio_path = os.path.join(audio_dir, os.path.splitext(filename)[0] + '.wav')
            command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path}"
            subprocess.call(command, shell=True)

def extract_vggish_features(audios, srs):
    features = []
    for audio, sr in zip(audios, srs):
        # Convert audio to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Get non-silent segments
        non_silent_intervals = librosa.effects.split(audio, top_db=20)
        non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
        
        # Resample to 16kHz
        audio_resampled = librosa.resample(non_silent_audio, orig_sr=sr, target_sr=16000)
        
        # Ensure the audio length is at least 1 second (16,000 samples)
        if len(audio_resampled) < 16000:
            audio_resampled = np.pad(audio_resampled, (0, 16000 - len(audio_resampled)), mode='constant')
        else:
            audio_resampled = audio_resampled[:16000]

        # Convert audio to 128-dimensional embedding using VGGish
        embedding = vggish_model(audio_resampled)
        features.append(embedding.numpy().flatten())
    return np.array(features)

# Function to load audio files and labels
def load_audio_files(audio_dir, label):
    audios = []
    srs = []
    labels = []
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(audio_dir, filename)
            y, sr = librosa.load(filepath, sr=None)
            audios.append(y)
            srs.append(sr)
            labels.append(label)
    return audios, srs, labels

if __name__=='__main__':
    # Define paths
    viral_video_dir = 'clips/Viral'
    non_viral_video_dir = 'clips/NonViral'
    viral_audio_dir = 'Viral_Audio'
    non_viral_audio_dir = 'NonViral_Audio'

    # Function to extract audio from videos


    extract_audio_from_videos(viral_video_dir, viral_audio_dir)
    extract_audio_from_videos(non_viral_video_dir, non_viral_audio_dir)


    # Load VGGish model
    vggish_model = hub.load('https://tfhub.dev/google/vggish/1')

    # Function to extract non-silent segments and VGGish features

    # Load data
    viral_audios, viral_srs, viral_labels = load_audio_files(viral_audio_dir, 1)
    non_viral_audios, non_viral_srs, non_viral_labels = load_audio_files(non_viral_audio_dir, 0)

    all_audios = viral_audios + non_viral_audios
    all_srs = viral_srs + non_viral_srs
    all_labels = viral_labels + non_viral_labels

    # Split the data into training and testing sets
    X_train, X_test, sr_train, sr_test, y_train, y_test = train_test_split(all_audios, all_srs, all_labels, test_size=0.2, random_state=42)

    # Convert labels to categorical
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    # Extract features
    X_train_features = extract_vggish_features(X_train, sr_train)
    X_test_features = extract_vggish_features(X_test, sr_test)

    # Normalize features
    scaler = StandardScaler()
    X_train_features = scaler.fit_transform(X_train_features)
    X_test_features = scaler.transform(X_test_features)

    # Save the scaler
    scaler_path = 'scaler.pkl'
    joblib.dump(scaler, scaler_path)



    # Build and train model
    classification_model = build_classification_model(X_train_features.shape[1])
    history = classification_model.fit(X_train_features, y_train, epochs=50, batch_size=16, validation_split=0.2)

    # Evaluate model
    loss, accuracy = classification_model.evaluate(X_test_features, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    classification_model.save_weights('model_sound.weights.h5')