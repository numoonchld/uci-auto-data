#cython: language_level=3

# import ml libraries 
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

import tensorflow as tf
import keras

import pickle

### IMPORTS: ==================================================================

## unsplit data:
data_path = '../csv/00-cleaned-up-data.csv'
df = pd.read_csv(data_path)

## training data:
x_train = pd.read_csv('../csv/02-x_train.csv')
y_train = pd.read_csv('../csv/02-y_train.csv')

## test data:
x_test = pd.read_csv('../csv/02-x_test.csv')
y_test = pd.read_csv('../csv/02-y_test.csv')

### PRE-PROCESSING: ===========================================================

# check for nas:
df.isna().sum()

# one-hot all categorical variables:
df.columns

# Index(['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
#        'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
#        'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
#        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
#        'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
#        'highway-mpg', 'price'],
#       dtype='object')

## categorical variables:
# symboling
# make
# fuel-type
# aspiration
# num-of-doors
# body-style
# drive-wheels
# engine-location
# engine-type
# number-of-cylinders
# fuel-system

ohe = pd.get_dummies(df)
ohe = pd.get_dummies(ohe, columns=['symboling'])

### TEST-TRAIN SPLIT: ==========================================================

train_data = ohe.sample(frac=0.8,random_state=0)
test_data = ohe.drop(train_data.index)

### CONSOLE LOG DATA: ==========================================================

# print(train_data.columns)
# print(train_data.tail())

### PROCESS DATA: ==============================================================

train_target = train_data.pop('price')
test_target = test_data.pop('price')

train_stats = train_data.describe().transpose()
# print(train_stats)

### NORMALIZE DATA: ============================================================

normed_train_data = (train_data - train_stats['mean']) / train_stats['std']
normed_test_data = (test_data - train_stats['mean']) / train_stats['std']

# print(normed_train_data)
# print(normed_test_data)

### KERAS MODEL: ===============================================================

# print(len(normed_train_data.keys()))
# print(normed_train_data.shape)

# print(len(normed_train_data.columns))
# print(normed_train_data.columns)

## narrowed-down features: --------------------------------------------------------

# kliene = normed_train_data[['engine-size', 'bore', 'stroke', 'compression-ratio']]
# essent = normed_train_data[['engine-size', 'bore', 'stroke', 'compression-ratio']]
# essent_plus = normed_train_data[['engine-size', 'bore', 'stroke', 'compression-ratio']]

# setup MLP model generating function
cpdef build_mlp_model():

    model = keras.Sequential([
        keras.layers.Dense(1024, activation = 'sigmoid', input_dim = len(normed_train_data.keys())),
        keras.layers.Dense(512, activation = 'sigmoid'),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mse',
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001, nesterov=True),
                  metrics=['mae', 'mse'])

    return model


model = build_mlp_model()
# model.summary()

## model verification: ----------------------------------------------------------

# example_batch = kliene[:10]
example_batch = normed_train_data[:10]
# print(example_batch)
example_result = model.predict(example_batch)
print(example_result)

### TRAIN MODEL: ====================================================================

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    print('.', end = '')
    if epoch % 100 == 0: print('\nEPOCH: ', epoch, '\n')
    

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_target,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
