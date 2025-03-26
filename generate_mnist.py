#!/usr/bin/env python3

import pandas as pd

dataframe = pd.read_csv("mnist_train.csv", header=None)
labels = dataframe.iloc[:, 0]
dataframe = dataframe.iloc[:, 1:] / 255.0

for i in range(10):
    dataframe[f'label_{i}'] = (labels == i).astype(int)

dataframe.to_csv("mnist_train_one_hot.csv", index=False, header=False)