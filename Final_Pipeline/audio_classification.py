from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from sklearn.preprocessing import StandardScaler
import joblib
import librosa

# Define your model architecture (same as the one used during training)
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

def preprocess_audio(audio_path,vggish_model,scaler):
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Convert to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to 16kHz
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # Ensure the audio length is at least 1 second (16,000 samples)
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)), mode='constant')
    else:
        audio = audio[:16000]
    
    # Convert audio to 128-dimensional embedding using VGGish
    embedding = vggish_model(audio)
    features = embedding.numpy().flatten()
    
    # Normalize features using the pre-fitted scaler
    features = scaler.transform([features])[0]
    
    return features

def predict_viral(audio_path):
    model,vggish_model,scaler=setup()
    features = preprocess_audio(audio_path,vggish_model,scaler)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    predictions = model.predict(features)
    return predictions

def setup():### Change name
    model=build_classification_model(128)
    # Load the saved weights
    # Load VGGish model
    vggish_model = hub.load('https://tfhub.dev/google/vggish/1')

    # StandardScaler instance to normalize features (fit it on training data)
    scaler_path = 'scaler.pkl'
    scaler = joblib.load(scaler_path)

    model.load_weights('model_sound3.weights.h5')

    # Compile the model (same as during training)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model,vggish_model,scaler


# # Example usage
# audio_path = 'clips/clip_22.wav'
# result = predict_viral(audio_path)
# print(f'The audio is: {result}')
