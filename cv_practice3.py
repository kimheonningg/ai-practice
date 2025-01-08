"""
CNN - Convolutional Neural Network

CNN is a filter used to extract specific features from the data.

It uses convolution (element-wise multiplication) to make the feature map.


Pooling

Pooling is a method of reducing data size that conserves the meaning of the data.

(Ex, 16 * 16 pixel image -> 2 * 2 image)

Ex. Max Pooling - divides the data into smaller areas, and extract the max value from each area.
"""

# using CNN

import tensorflow as tf

data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
# convert the data to size 28 * 28 * 1
# 60000 images are included in training_images
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
# converting data size (same as above)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu',
                           input_shape = (28, 28, 1)),
    # number of filters: 64,
    # filter size: 3 * 3 (is the most used size, normally use odd numbers for filter size)
    # image size is 28 * 28 (1 for the 3rd dimension)
    tf.keras.layers.MaxPooling2D((2, 2)),
    # divide the image into smaller parts with sizes of 2 * 2
    # choose the max number in the 2 * 2 sized part
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(training_images, training_labels, epochs = 5) 
# epochs = 25 takes too long

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

print(model.summary())