# Fruit Image Classification

This project focuses on image classification using a MobileNetV2 convolutional neural network (CNN) to classify 35 different types of fruit. Please note that this code snippet represents only a section of a larger project that encompasses various components, as described below.

## Project Overview

The aim of this project is to develop a comprehensive fruit classification system using deep learning techniques. The specific tasks include:

Training a MobileNetV2 CNN model to classify 35 different types of fruit.
Utilizing the Keras framework for deep learning implementation.
Preprocessing the fruit images to ensure optimal model performance.
Evaluating the trained model's accuracy and performance on test datasets.
Providing predictions for fruit types based on input images.
Please be aware that this README file focuses on the image classification component of the project. The complete project entails additional functionalities and features, as outlined in the project description.

## Dataset
The fruit classification system is trained on a carefully curated dataset consisting of high-quality images of 35 different fruit types. Each fruit category contains a diverse set of images captured under various conditions, angles, and backgrounds. The dataset aims to ensure the model's robustness and generalization to real-world scenarios.

## Model Architecture
The MobileNetV2 architecture serves as the backbone for this image classification project. This model is specifically chosen due to its efficiency, versatility, and the ability to handle datasets with limited training images. The MobileNetV2 CNN model provides an optimal balance between accuracy and computational efficiency, making it suitable for real-time fruit classification.

## Usage
To use the image classification functionality of this project, please follow these steps:

Ensure that the required dependencies, including Python, Keras, tensorflow, and relevant libraries, are installed on your system.

Ensure your computer has a detectable webcam 

Ensure hdf5 model, labels.json, and Fruit.png all have the correct file path to your directory.

```
camera.export_to_png('Fruit.png')
``` 

Navigate to the appropriate directory where the image classification code resides.

```
run main.py
```

Run the provided code snippet to initiate the image classification process.

Analyze the output to obtain the predicted fruit type(s) for the given input image(s).

### Please note that this code snippet serves as a part of a larger project. For a comprehensive understanding of the entire project and access to all its features, please refer to the complete project repository.

## Acknowledgments

I would like to express my gratitude to the creators and contributors of the original datasets used in this project. Their efforts in collecting and organizing the fruit images have been instrumental in training the accurate fruit classification model.

## Contact
For any questions, concerns, or collaboration opportunities related to this project, please feel free to reach out to diaz469@csusm.edu
