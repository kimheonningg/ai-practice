# !pip install tensorflow
import tensorflow as tf
# print(tf.__version__)

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(units=1, input_shape=[1])])
# model with 1 Dense layer with 1 neuron ("units=1"),
# input data size is 1 ("input_shape=[1]")

model.compile(optimizer='sgd', loss='mean_squared_error')
# use sgd-stochastic gradient descent- optimizer

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
# sample data

model.fit(xs, ys, epochs=500)
# train 500 times

print(model.predict(np.array([10.0])))