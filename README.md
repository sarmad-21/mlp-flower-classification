# Flower Classification with MLP 

## Overview 
This project aims to classify flower images into two categories based on their labels using a Multi-Layer Perceptron (MLP) Classifier. The dataset includes images labeled as "0" and "5". The classifier is trained to distinguish between these two categories.

## Technologies Used 
- Python
- Scikit-Learn
- Numpy
- PIL (Python Imaging Library)

## Deep Learning Model 
The model uses a Multi-Layer Perceptron (MLP) architecture to classify flower images.
- **Model Architecture**:
  - One hidden layer with 100 neurons
  - ReLU activation function
  - Output layer with sigmoid activation for binary classification
- **Compilation**:
  - Optimizer: Adam
  - Loss Function: Binary cross-entropy
- **Training**:
  - Number of epochs: 1000
  - Batch size: Not explicitly defined (handled internally by `MLPClassifier`)
- **Performance**:
  - Achieved a validation accuracy of approximately 70%.

## Data 
- The dataset consists of images of flowers and their corresponding labels.
- Labels are provided in the `flower_labels.csv` file.
- Images are resized to 32x32 pixels for processing.

## Future Enhancements 
- Later used a Convolutional Neural Network (CNN) to achieve higher accuracy in flower classification which is included in my cnn-flower-classification repository. 
