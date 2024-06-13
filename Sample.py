import cv2
import numpy as np
import tensorflow as tf
import urllib.request

# Load the TensorFlow model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# Download an example image, any image link can be used here
image_url = "You can find any picture from any website, paste the link here"
urllib.request.urlretrieve(image_url, "example_image.jpg")

# Load the image and check for errors
image_path = "example_image.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Image could not be loaded. Check the file path: {}".format(image_path))

# Resize and preprocess the image
image = cv2.resize(image, (224, 224))
image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

# Resize the image for model input
input_image = np.expand_dims(image, axis=0)

# Perform object detection on the image
predictions = model.predict(input_image)

# Find the top predicted class
predicted_class = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]

# Print the result
print("Predicted object:", predicted_class[1])
print("Prediction confidence:", predicted_class[2])
