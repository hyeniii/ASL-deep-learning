# Image Recognition with Custom YOLO

This project contains an implementation of a custom version of the YOLO (You Only Look Once) algorithm for multi-object detection and classification. The main goal of this implementation is to be able to recognize and classify images from a dataset of annotated images. It uses a modified YOLO architecture for object detection in images, and TensorFlow's Keras API for training the model.

## Description

The ipynb file contains code that:
1. Preprocesses images and bounding box coordinates from a given dataset.
2. Defines a custom YOLO-like model architecture.
3. Defines custom loss functions for bounding box regression and objectness and class predictions.
4. Trains the model with the custom loss function.
5. Saves the trasined model.
6. Loads the trained model.
7. Tests the model's performance on a test set, and computes a confusion matrix.
8. Performs inference on an individual image file.

## Requirements

* Python 3.7 or above
* TensorFlow 2.4 or above
* Keras 2.4.3
* PIL
* Seaborn
* Matplotlib
* Scikit-learn

## Dataset Structure

The dataset should be structured in the following format:
```bash
data_tf
├── test
│   ├── A
│   ├── ...
│   ├── Z
│   └── _annotations.csv
├── train
│   ├── A
│   ├── ...
│   ├── Z
│   └── _annotations.csv
└── valid
│   ├── A
│   ├── ...
│   ├── Z
│   └── _annotations.csv
```

Please note that the model expects images to be in `.jpg` format.

