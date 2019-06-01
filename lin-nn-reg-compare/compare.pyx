
# cython: language_level=3

### IMPORT ML LIBRARIES: ======================================================
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
import keras
#from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import cross_val_predict

### IMPORTS: ==================================================================

data_path = '../csv/00-cleaned-up-data.csv'
df = pd.read_csv(data_path)

### FEATURE SELECTION: ========================================================

group = np.array([
    'engine-size', 'horsepower', 'city-mpg', 'highway-mpg', 'body-style',
    'price'
])

df = df[group]

### ONE-HOT-ENCODING: ==========================================================

ohe = pd.get_dummies(df)
print(ohe.shape)

### TEST-TRAIN SPLIT: ==========================================================

train_data = ohe.sample(frac=0.8, random_state=0)
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

### REGRESSION MODELS: ========================================================

## MULTI-LINEAR REGRESSION: --------------------------------------

# init model:
lm = LinearRegression()

# train model:
lm.fit(normed_train_data, train_target)

## NEURAL NET REGRESSION: ----------------------------------------


# setup MLP model generating function
def build_mlp_model():

    model = keras.Sequential([
        keras.layers.Dense(
            32,
            activation='sigmoid',
            kernel_initializer=keras.initializers.glorot_normal(seed=3),
            input_dim=len(normed_train_data.keys())),
        keras.layers.Dropout(rate=0.25, noise_shape=None, seed=7),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dropout(rate=0.001, noise_shape=None, seed=3),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(lr=0.09,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  epsilon=None,
                                                  decay=0.03,
                                                  amsgrad=True),
                  metrics=['mae', 'mse'])

    return model


# initialize model and view details

nn = build_mlp_model() # keras only model
nn.summary()

# nn model verification: ----------------------------------------------------------

example_batch = normed_train_data[:10]
example_result = nn.predict(example_batch)
print(example_result)

# train model: --------------------------------------------------------------------


# training progress display funtion:
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print('.', end='')
        if epoch % 100 == 0: print('\nEPOCH: ', epoch, '\n')

EPOCHS = 1000

nn.fit(normed_train_data,
       train_target,
       epochs=EPOCHS,
       validation_split=0.2,
       verbose=0,
       callbacks=[PrintDot()])

### RESIDUALS (in-sample accuracy): ===========================================

## MULTI-LINEAR REGRESSION: --------------------------------------
plt.figure(0, figsize=(12, 10))
sns.residplot(lm.predict(normed_train_data), train_target)
plt.savefig('plots/a-resid-plot-multi-lin.png')

## NEURAL NET REGRESSION: ----------------------------------------
plt.figure(1, figsize=(12, 10))
sns.residplot(nn.predict(normed_train_data).flatten(), train_target)
plt.savefig('plots/a-resid-plot-nn.png')

### ACCURACY (out-of-sample test): ============================================

## MULTI-LINEAR REGRESSION: --------------------------------------

print('\nMulti-variate Linear Regression Accuracy Metrics')

# R^2 score:
print('out-of-sample R^2')
print(r2_score(test_target, lm.predict(normed_test_data)))

# MSE score:
print('out-of-sample MSE')
print(mean_squared_error(test_target, lm.predict(normed_test_data)))

## NEURAL NET REGRESSION: ----------------------------------------

print('\nNeural Net Regression Accuracy Metrics')

# R^2 score:
print('out-of-sample R^2')
print(r2_score(test_target, nn.predict(normed_test_data).flatten()))

# MSE score:
print('out-of-sample MSE')
print(mean_squared_error(test_target, nn.predict(normed_test_data).flatten()))

## DISTRIBUTION PLOTS: -------------------------------------------

plt.figure(2, figsize=(12, 10))

ax1 = sns.distplot(test_target, hist=False, color="k", label='True Values')

ax2 = sns.distplot(lm.predict(normed_test_data),
                   hist=False,
                   color="c",
                   label='Linear Model Prediction',
                   ax=ax1)

ax3 = sns.distplot(nn.predict(normed_test_data).flatten(),
                   hist=False,
                   color="y",
                   label='Neural Net Prediction',
                   ax=ax1)

plt.title(
    "['engine-size', 'horsepower', 'city-mpg', 'highway-mpg', 'body-style']:Price"
)
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.savefig('plots/b-dist-lin-nn-compare.png')
