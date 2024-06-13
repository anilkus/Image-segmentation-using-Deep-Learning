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

      ![image](https://github.com/anilkus/Image-segmentation-using-Deep-Learning/assets/16832969/d90e0bca-8892-4825-8d2c-c6993d340391)


