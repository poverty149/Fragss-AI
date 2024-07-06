# Action Recognition Model using Transfer Learning with C3D Architecture

This project demonstrates the implementation of an action recognition model using transfer learning on the C3D (Convolutional 3D) architecture. The pretrained weights used in this model were originally trained on the Sports-1M dataset. Here's how the model was constructed and trained using a custom dataset with new labels.

## Model Architecture

The model architecture used is based on the C3D architecture, which is designed for video analysis tasks. The C3D model consists of convolutional layers followed by max pooling and fully connected layers. Here's a summary of the architecture used:

- **Input Shape**: `(16, 112, 112, 3)` - 16 frames of size 112x112 pixels with 3 color channels (RGB).
- **Layers**:
  - Convolutional Layers with ReLU activation.
  - MaxPooling3D layers for spatial pooling.
  - Fully Connected (Dense) layers at the end for classification.

## Transfer Learning Approach

Transfer learning was applied by leveraging pretrained weights from a C3D model trained on the Sports-1M dataset. The pretrained model's weights were loaded into the C3D architecture, except for the final classification layer, which was adapted to suit the new action categories of interest.

## Custom Dataset

The custom dataset used in this project consists of videos categorized into two classes: "Headshots" and "Grenades". These videos were collected and labeled specifically for training an action recognition model in gaming videos.

### Dataset Preparation

1. **Data Collection**: Videos were collected from youtube where actions like "Headshots" and "Grenades" were performed.
   
2. **Labeling**: Videos were manually labeled into two categories based on the action performed.

3. **Data Processing**:
   - Videos were resized to `(112, 112)` pixels to match the input size expected by the C3D model.
   - Each video was divided into chunks of 16 frames with an overlap of 8 frames for feature extraction.

## Training

### Feature Extraction

- **Feature Extraction**: Video frames were processed using the pretrained C3D model to extract high-level features. These features were averaged and normalized before being fed into the classification layers.

### Model Training

- **Training Procedure**:
  - The model was compiled with the Adam optimizer and categorical cross-entropy loss function.
  - Early stopping was implemented to monitor validation loss and prevent overfitting.

- **Training Results**: The model was trained on a portion of the dataset and evaluated on a validation set to ensure generalization.

## Usage

### Predictions

- **Prediction Pipeline**: After training, the model can predict actions (Headshots or Grenades) from new gaming videos by extracting features and classifying them using the trained model.

### Additional Info
Download the weights for the pretrained model from the link [C3D Sports 1M weights](https://drive.google.com/file/d/1rlZ-xTkTMjgWKiQFUedRnHlDgQwx6yTm/view)

Will soon post a link for the weights obtained on our custom gaming data that contains headshots and grenades. We can modify the code later to account for other classes by collecting videos pertaining to them.

