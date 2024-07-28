# Image Processing Project

This project focuses on image processing and classification using a Convolutional Neural Network (CNN) model built with TensorFlow and Keras. The project is divided into three tasks:

Task 1: Preprocessing and filtering images for relevance.
Task 2: Building, training, and saving a CNN model for image classification.
Task 3: Loading the trained model and using it to classify new images.

## Code Structure

Task 1: Preprocessing and Filtering Images
This task involves loading, preprocessing, and filtering any uploaded images to check that only relevant images are used for training.

Key steps:
* relevant_image(img): Checks if an image is relevant by verifying its format.
* for loop: Iterates through the image directory, removes irrelevant images, and preprocesses the relevant ones.



Task 2: Building and Training the CNN Model
This task involves creating, training, and saving a CNN model for image classification.

Key Steps:
* Model Definition: Defines a CNN model with convolutional, pooling, flatten, dropout, and dense layers.
* Model Compilation: Compiles the model using the Adam optimiser and categorical crossentropy loss function.
* Model Training: Trains the model using the preprocessed images.
* Model Saving: Saves the model for future use.


Task 3: Loading and Using the Trained Model
This task involves loading the saved model and using it to classify new images.

Key steps:
* classify_image(image_path): Loads and preprocesses an image, predicts its class using the trained model, and returns the predicted class name.



## Usage
Requirements:
* Python installed on your computer.
* Required libraries installed:
  *   OpenCV
  *   Matplotlib
  *   TensorFlow and Keras
  *   NumPy


