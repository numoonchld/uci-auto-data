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

### NARROW DOWN FEATURES: ======================================================
# thsi dataset has a total of 26 features, 11 are categorical; after one-hot encoding the categorical ones, it makes a total of 69 predictor variables to predict the target (price) variable
# the dataset contains 159 observations, and 69 features (after one-hot-ecoding) might be a very small dataset to train a deep neural model as the pridications are all NaN outputs

# so only a few features will be used at a time to train models; 
# the correlation charts are consulted, and only the relevant predictors are chosen

## group 1: 
group1 = np.array(['engine-size','horsepower','city-mpg','highway-mpg','body-style','price'])

df = df[group1]
# print(df.keys())

# apply 'one-hot-encoding": --------------------------------------------------

ohe = pd.get_dummies(df)
#ohe = pd.get_dummies(ohe, columns=['symboling'])
# print(ohe.keys())


### TEST-TRAIN SPLIT: ==========================================================

train_data = ohe.sample(frac=0.8,random_state=0)
test_data = ohe.drop(train_data.index)

### NORMALIZE DATA: ============================================================

train_target = train_data.pop('price')
test_target = test_data.pop('price')

train_stats = train_data.describe().transpose()
# print(train_stats)

normed_train_data = (train_data - train_stats['mean']) / train_stats['std']
normed_test_data = (test_data - train_stats['mean']) / train_stats['std']

# print(normed_train_data.dtypes)
# print(normed_test_data)

### KERAS MODEL: ===============================================================

# setup MLP model generating function
def build_mlp_model():

    model = keras.Sequential([
        # keras.layers.Dense(64, activation = 'sigmoid', kernel_initializer=keras.initializers.RandomNormal(mean=0.001, stddev=0.05, seed=1), input_dim = len(normed_train_data.keys())),
        keras.layers.Dense(64, activation = 'sigmoid', kernel_initializer=keras.initializers.glorot_normal(seed=3), input_dim = len(normed_train_data.keys())),
        # keras.layers.Dense(64, activation = 'sigmoid', kernel_initializer=keras.initializers.RandomNormal(mean=0.001, stddev=0.05, seed=1), input_dim = len(normed_train_data.keys())),
        keras.layers.Dense(8, activation = 'relu'),
        keras.layers.Dense(1)
    ])

    model.compile(
                    loss = 'mse',
                    #optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=True),
                    optimizer=keras.optimizers.Adam(lr=0.09, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.03, amsgrad=True),
                    #optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                    #optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
                    #optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
                    metrics=['mae', 'mse']
                    )

    return model

# initialize model and view details
model = build_mlp_model()
model.summary()

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

## visualize learning

# extract learning from the fit output:
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

# plot the learning steps: 
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure(0)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [USD]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  # plt.ylim([2000,50000])
  plt.legend()
  
  plt.figure(1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$USD^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  # plt.ylim([2000,50000])
  plt.legend()
  plt.show()

plot_history(history)


### TEST SET EVALUATION: =====================================================

loss, mae, mse = model.evaluate(normed_test_data, test_target, verbose=0)
print("\n\nTesting set Mean Abs Error: {:5.2f} USD".format(mae))

### PREDICTIONS: =============================================================

test_predictions = model.predict(normed_test_data).flatten()

# plot scatter plot for test data -----------------------

plt.figure(2)
plt.scatter(test_target, test_predictions)
plt.xlabel('True Values [USD]')
plt.ylabel('Predictions [USD]')
plt.axis('equal')
plt.axis('square')
# plt.xlim([0,plt.xlim()[1]])
# plt.ylim([0,plt.ylim()[1]])
# _ = plt.plot([-100, 100], [-100, 100])

# error distribution plot: ------------------------------
plt.figure(3)
error = test_predictions - test_target
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [USD]")
_ = plt.ylabel("Count")
plt.show()