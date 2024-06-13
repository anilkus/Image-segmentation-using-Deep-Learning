# Image-segmentation-using-Deep-Learning
Image segmentation using A convolutional neural network (CNN) model is defined using the Sequential API from TensorFlow.


Explanation:

Data Loading: The CIFAR-10 dataset is loaded and a subset of 256 training images and labels is selected.

Normalization: The pixel values of the images are normalized to the range [0, 1].

Model Creation: A convolutional neural network (CNN) model is defined using the Sequential API from TensorFlow.
The model consists of:

Three convolutional layers with sigmoid activation functions.
Two max-pooling layers.
A flattening layer to convert 2D features to 1D.
Two dense (fully connected) layers with the final layer having 10 units (for the 10 classes in CIFAR-10).

Model Compilation: The model is compiled with an SGD optimizer (learning rate set to 0.9), sparse categorical cross-entropy loss function, and accuracy as the metric.

Model Summary: A summary of the model architecture is printed.

Model Training: The model is trained for 10 epochs on the training data, with validation on the test data.

Visualization: The training and validation accuracy over epochs are plotted to visualize the model's performance.

![image](https://github.com/anilkus/Image-segmentation-using-Deep-Learning/assets/16832969/486bd995-4911-4ecb-a1b8-59b47a972725)

---------
Here is an explanation of the 2nd part of the project:

Imports and Model Loading: The necessary libraries (cv2, numpy, tensorflow, and urllib) are imported. The MobileNetV2 model, pre-trained on the ImageNet dataset, is loaded with the top classification layer included.
Image Download: An example image is downloaded from the provided URL and saved as "example_image.jpg".

Image Loading and Error Checking: The image is loaded using OpenCV (cv2.imread). If the image cannot be loaded, a FileNotFoundError is raised.

Image Preprocessing: The image is resized to 224x224 pixels, which is the required input size for MobileNetV2. The image is then preprocessed using the preprocess_input function specific to MobileNetV2.

Model Input Preparation: The image is expanded to include a batch dimension, making it suitable for model input.

Object Detection: The preprocessed image is passed through the model to obtain predictions.

Prediction Decoding: The top predicted class is decoded from the model's predictions using decode_predictions.

Result Printing: The predicted object class and the confidence of the prediction are printed.


