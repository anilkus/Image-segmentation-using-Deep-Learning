import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Loaded the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = train_images[:256]
train_labels = train_labels[:256]

# Normalization step
train_images, test_images = train_images / 255.0, test_images / 255.0

# Created the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='sigmoid'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='sigmoid'),
    layers.Flatten(),
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(10)
])

# Compiled the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.9)  # Set learning rate here
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model summary
model.summary()

# Trained the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Visualized the training results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
