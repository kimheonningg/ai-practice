# hard coding epochs

# 28 * 28 size grid
# pixel value 0 ~ 255

import tensorflow as tf

data = tf.keras.datasets.fashion_mnist # import fashion MNIST data

(training_images, training_labels), (test_images, test_labels) = data.load_data()
# image: 28 * 28 grid containing pixels with values of 0 ~ 255
# label: 0 ~ 9 (10 categories)

training_images = training_images / 255.0
test_images = test_images / 255.0
# normalizing data have values of 0 ~ 1
# normalization increases model performance

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # input: 28 * 28 array
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    # hidden layer with 128 neurons- 128 is an arbitrary value
    # activation function: ReLU
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    # output layer with 10 neurons- because there are 10 categories
    # activation function: softmax
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(training_images, training_labels, epochs = 5)

# test with test data
model.evaluate(test_images, test_labels)

# check if the model predicts well
classifications = model.predict(test_images)
print(classifications[0])
# prints array with length 10
# probability of index of test_labels[0] should be the highest in classifications[0] array
print(test_labels[0])