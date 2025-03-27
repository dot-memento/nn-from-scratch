#!/usr/bin/env python3

import pandas as pd
from keras.datasets import mnist

print("Loading MNIST dataset from keras...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 784)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 784)
x_test = x_test.astype('float32') / 255.0

train_dataframe = pd.DataFrame(x_train)
test_dataframe = pd.DataFrame(x_test)

for i in range(10):
    train_dataframe[f'label_{i}'] = (y_train == i).astype(int)
    test_dataframe[f'label_{i}'] = (y_test == i).astype(int)

print("Saving to mnist_train_one_hot.csv...")
train_dataframe.to_csv("mnist_train_one_hot.csv", index=False, header=False)
print("Saving to mnist_test_one_hot.csv...")
test_dataframe.to_csv("mnist_test_one_hot.csv", index=False, header=False)
print("Done!")