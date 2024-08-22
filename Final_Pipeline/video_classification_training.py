import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# Constants
FRAME_RATE = 1  # Number of frames to extract per second of video
IMAGE_SIZE = (224, 224)  # Size to resize frames
BATCH_SIZE = 32
EPOCHS = 50
DATA_DIR = './clips'  # Directory containing 'Viral' and 'NonViral' folders

# Function to extract frames from video using OpenCV
def extract_frames(video_path, frame_rate=FRAME_RATE):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if success and count % int(vidcap.get(cv2.CAP_PROP_FPS) / frame_rate) == 0:
            image = cv2.resize(image, IMAGE_SIZE)
            frames.append(image)
        count += 1
    vidcap.release()
    return frames

# Load data
def load_data(data_dir):
    data = []
    labels = []
    for label in ['Viral', 'NonViral']:
        folder = os.path.join(data_dir, label)
        for video_file in os.listdir(folder):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(folder, video_file)
                frames = extract_frames(video_path)
                data.extend(frames)
                labels.extend([label] * len(frames))
    return np.array(data), np.array(labels)

 # Extract features using MobileNetV2
def extract_features(model, data):
    features = model.predict(data, batch_size=BATCH_SIZE, verbose=1)
    return features

if __name__=='__main__':
    images, labels = load_data(DATA_DIR)
    # Encode labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels).flatten()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Load MobileNetV2 model for feature extraction
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

   

    # Extract features for training and testing data
    X_train_features = extract_features(base_model, X_train)
    X_test_features = extract_features(base_model, X_test)

    # Flatten the features
    X_train_features = X_train_features.reshape((X_train_features.shape[0], -1))
    X_test_features = X_test_features.reshape((X_test_features.shape[0], -1))


    # Build and train the model
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(X_train_features.shape[1],)))
    model.add(Dense(1, activation='sigmoid'))



    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train_features, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test_features, y_test))

    # Save the trained model
    model.save('video_virality_model.h5')

    print("Model training complete and saved as 'video_virality_model.h5'")

