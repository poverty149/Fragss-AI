import os
import numpy as np
import cv2
from keras.models import  Sequential
from keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D, Flatten, Dense, Dropout
from keras.utils import to_categorical


# Create the C3D model using Sequential API


def create_model_sequential():
    model = Sequential()
    input_shape = (16, 112, 112, 3)

    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='conv1', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1'))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    return model

# Create the feature extractor using Sequential API
def create_sequential_feature_extractor(model, layer_name='fc6'):
    feature_extractor = Sequential()
    for layer in model.layers:
        feature_extractor.add(layer)
        if layer.name == layer_name:
            break
    return feature_extractor


# Load and preprocess videos
def load_and_preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))
        frame = frame.astype('float32') / 255.0  # Normalize the frame
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Extract features from a video
def extract_video_features(video_path, extractor):
    video_frames = load_and_preprocess_video(video_path)
    chunks = divide_video_chunks(video_frames)
    features = extractor.predict(chunks)
    avg_features = np.mean(features, axis=0)
    norm_features = avg_features / np.linalg.norm(avg_features)
    return norm_features

# Divide a video into 16-frame chunks with 8-frame overlaps
def divide_video_chunks(video_frames):
    chunks = []
    for i in range(0, len(video_frames) - 16 + 1, 8):
        chunk = video_frames[i:i+16]
        chunks.append(chunk)
    return np.array(chunks)


def load_videos_from_folder(folder, label, frame_size=(112, 112), chunk_size=16, overlap=8):
    videos = []
    labels = []
    for filename in os.listdir(folder):
        video_path = os.path.join(folder, filename)
        if not video_path.endswith(('.mp4', '.avi')):
            continue
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, frame_size)
            frame = frame.astype('float32') / 255.0  # Normalize the frame
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        
        # Divide video into chunks
        num_chunks = (len(frames) - chunk_size) // overlap + 1
        for i in range(num_chunks):
            chunk = frames[i * overlap : i * overlap + chunk_size]
            if len(chunk) == chunk_size:
                videos.append(chunk)
                labels.append(label)
    return np.array(videos), np.array(labels)


def create_dataset():
    x_train, y_train = [], []

    # Load headshots videos
    headshots_folder = './training_videos/headshots/'
    headshots_videos, headshots_labels = load_videos_from_folder(headshots_folder, label=0)
    x_train.extend(headshots_videos)
    y_train.extend(headshots_labels)

    # Load grenades videos
    grenades_folder = './training_videos/grenades/'
    grenades_videos, grenades_labels = load_videos_from_folder(grenades_folder, label=1)
    x_train.extend(grenades_videos)
    y_train.extend(grenades_labels)

    x_train = np.array(x_train)
    y_train = to_categorical(np.array(y_train), num_classes=2)

    return x_train, y_train

def classify_video(video_path, feature_extractor, model, threshold=0.8):
    video_frames = load_and_preprocess_video(video_path)
    chunks = divide_video_chunks(video_frames)
    features = feature_extractor.predict(chunks)
    avg_features = np.mean(features, axis=0)
    norm_features = avg_features / np.linalg.norm(avg_features)
    
    predictions = model.predict(np.expand_dims(norm_features, axis=0))
    labels = ['headshots', 'grenades']
    predicted_labels = []

    for i, confidence in enumerate(predictions[0]):
        if confidence > threshold:
            predicted_labels.append(labels[i])

    if not predicted_labels:
        return ['other']

    return predicted_labels