import numpy as np
import cv2
import tensorflow as tf

## Using MobileNetV2
def preprocess_input(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame)
    return frame
def predict_virality(video_path, feature_extraction_model, classification_model):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (224, 224)))
    cap.release()
    
    video_features = []
    for frame in frames:
        frame = preprocess_input(frame)
        frame_features = feature_extraction_model.predict(np.expand_dims(frame, axis=0))
        video_features.append(frame_features.flatten())
    video_features = np.mean(video_features, axis=0)
    
    prediction = classification_model.predict(np.expand_dims(video_features, axis=0))
    return prediction

# Example usage
# prediction = predict_virality('new_video.mp4', model, classification_model)
# print(f"Viral Probability: {prediction[0][1]}")
# print(f"Non-Viral Probability: {prediction[0][0]}")
