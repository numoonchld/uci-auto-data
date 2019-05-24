# import ml libraries 
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

import tensorflow as tf
import keras
from keras import layers

import pickle

### IMPORTS: ==================================================================

## unsplit data:
data_path = 'csv/00-cleaned-up-data.csv'
df = pd.read_csv(data_path)

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
print(train_stats)

### NORMALIZE DATA: ============================================================

normed_train_data = (train_data - train_stats['mean']) / train_stats['std']
normed_test_data = (test_data - train_stats['mean']) / train_stats['std']

### KERAS MODEL: ===============================================================

# Keras: The Sequential model is a linear stack of layers.

# setup model within a function 

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()
model.summary()

# verify model before training

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result