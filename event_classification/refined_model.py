## The below implementation incorporates a temporal convolution network with attention mechanism to the previous base model

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, Conv1D, Input, Layer, Permute, Reshape, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Define paths
train_dir = 'path/to/train_data'
val_dir = 'path/to/val_data'
num_classes = len(os.listdir(train_dir))  # Number of event classes
batch_size = 32
img_height, img_width = 224, 224
sequence_length = 10  # Number of frames per video

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Extract features using VGG16
def extract_features(generator, num_samples, batch_size, sequence_length):
    features = np.zeros((num_samples, sequence_length, 7, 7, 512))  # Adjust based on output shape of VGG16
    labels = []
    i = 0

    for inputs_batch, labels_batch in generator:
        for j in range(sequence_length):
            if i >= num_samples:
                break
            inputs = inputs_batch[j::sequence_length]
            features_batch = base_model.predict(inputs)
            features[i] = features_batch
            labels.append(labels_batch[j])
            i += 1
        if i >= num_samples:
            break

    return features, np.array(labels)

num_train_samples = len(train_generator.filenames) // sequence_length
num_val_samples = len(val_generator.filenames) // sequence_length

train_features, train_labels = extract_features(train_generator, num_train_samples, batch_size, sequence_length)
val_features, val_labels = extract_features(val_generator, num_val_samples, batch_size, sequence_length)

# Label encoding
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
val_labels = le.transform(val_labels)

train_labels = to_categorical(train_labels, num_classes)
val_labels = to_categorical(val_labels, num_classes)

# Temporal Convolutional Network (TCN) with Attention Mechanism
def attention_3d_block(inputs):
    input_dim = int(inputs.shape[-1])
    a = Dense(input_dim, activation='softmax')(inputs)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

input_shape = (sequence_length, 7, 7, 512)
inputs = Input(shape=input_shape)
x = TimeDistributed(Flatten())(inputs)
x = TimeDistributed(Dense(512, activation='relu'))(x)
x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
x = attention_3d_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile the model
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    train_features, train_labels,
    epochs=50,
    validation_data=(val_features, val_labels),
    batch_size=batch_size,
    callbacks=[checkpoint, early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(val_features, val_labels)
print(f'Validation Accuracy: {accuracy:.2f}')

# Save the final model
model.save('event_classification_model.h5')
