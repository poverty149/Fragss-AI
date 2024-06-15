import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the combined data
combined_df = pd.read_csv('combined_data.csv')

# Define the target variable and drop it from the dataset
scaler = StandardScaler()
combined_df[['views', 'likes', 'duration']] = scaler.fit_transform(combined_df[['views', 'likes',  'duration']])

# Define weights for each metric (subjective example)
weights = {
    'views': 0.5,
    'likes': 0.4,
    'duration': 0.1,
}

# Calculate virality score. Adjust this calculation later.
combined_df['virality_score'] = (
    combined_df['views'] * weights['views'] +
    combined_df['likes'] * weights['likes'] +
    combined_df['duration'] * weights['duration']
) * 100
target = combined_df['virality_score']
data = combined_df.drop(columns=['virality_score'])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42)

# Separate metadata and video features
metadata_features = [col for col in data.columns if not col.startswith('video_feature')]
video_features = [col for col in data.columns if col.startswith('video_feature')]

# Standardize metadata features
scaler = StandardScaler()
x_train_metadata = scaler.fit_transform(x_train[metadata_features])
x_test_metadata = scaler.transform(x_test[metadata_features])
x_train_video = x_train[video_features].values
x_test_video = x_test[video_features].values

# Define the metadata input model
metadata_input = Input(shape=(x_train_metadata.shape[1],), name='metadata_input')
metadata_dense = Dense(64, activation='relu')(metadata_input)
metadata_output = Dense(32, activation='relu')(metadata_dense)

# Define the video input model
video_input = Input(shape=(x_train_video.shape[1],), name='video_input')
video_dense = Dense(128, activation='relu')(video_input)
video_output = Dense(64, activation='relu')(video_dense)

# Concatenate video and metadata features
combined = Concatenate()([metadata_output, video_output])

# Final classification layer
output = Dense(1, activation='sigmoid')(combined)

# Define the model
model = Model(inputs=[metadata_input, video_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    [x_train_metadata, x_train_video],
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# Evaluate the model
y_pred = (model.predict([x_test_metadata, x_test_video]) > 0.5).astype("int32")
print(classification_report(y_test, y_pred, target_names=['class 0', 'class 1']))

# Save the trained model
model.save("multimodal_model.h5")
