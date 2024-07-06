import tensorflow as tf
import numpy as np
from keras.layers import  Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from utils import *




pretrained_weights_path = 'C3D_Sport1M_weights_keras_2.2.4.h5'
# Load pre-trained weights
c3d_model = create_model_sequential()
c3d_model.load_weights(pretrained_weights_path)
for layer in c3d_model.layers[:-3]:
    layer.trainable = False
# Compile the model to initialize it
c3d_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')

feature_extractor = create_sequential_feature_extractor(c3d_model, 'fc6')

# Test the feature extractor with dummy data
dummy_data = np.zeros((32, 16, 112, 112, 3), dtype='float32')
keras_tensor = tf.convert_to_tensor(dummy_data)

# Predict using the feature extractor
features = feature_extractor.predict(keras_tensor)
print(features.shape)

# Create the dataset
x_train, y_train = create_dataset()

# Check the shape of the dataset
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
num_classes = 2 # Change this to the number of classes in your dataset
c3d_model.pop()
c3d_model.add(Dense(num_classes, activation='softmax', name='fc8'))

# Compile the model
c3d_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
c3d_model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping])

# Example usage
video_path = './test_video.mp4'
predicted_class = classify_video(video_path, feature_extractor, c3d_model, threshold=0.8)
## Alternatively we can later adjust the classify videos function to have different thresholds for different classes
print(f'Predicted class for the video: {predicted_class}')
