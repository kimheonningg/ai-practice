# using TensorFlow DataSets (TFDS)
#!pip install tensorflow-datasets

import tensorflow as tf
import tensorflow_datasets as tfds

# method 1
# load all (train & test data) from fashion_mnist
mnist_data = tfds.load("fashion_mnist")

for item in mnist_data:
  print(item) # train data & test data

# method 2
# load train data only from fashion_mnist
mnist_train = tfds.load(name="fashion_mnist", split="train")
assert isinstance(mnist_train, tf.data.Dataset)
# understanding the structure of the data
print(type(mnist_train)) # PrefetchDataset object

for item in mnist_train.take(1):
  # get the first record of the PrefetchDataset object
  print(type(item)) # dictionary type
  print(item.keys()) # keys: 'item', 'label'
  print(item['image']) # 28 * 28 sized array
  print(item['label'])

# method 3
# load data with info
mnist_test, info = tfds.load(name = "fashion_mnist", with_info = "true")
print(info)


(training_images, training_labels), (test_images, test_labels) = \
  tfds.as_numpy(tfds.load('fashion_mnist', # convert objects into numpy arrays
                          split = ['train', 'test'],
                          batch_size = -1, # get all data
                          as_supervised = True)) # return as (image, label)

training_images = tf.cast(training_images, tf.float32) / 255.0
test_images = tf.cast(test_images, tf.float32) / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28, 1)),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs = 5)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])