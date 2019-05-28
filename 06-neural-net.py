# import ml libraries 
import numpy as np
import pandas as pd

import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt

import tensorflow as tf
import keras

import pickle

### IMPORTS: ==================================================================

## unsplit data:
data_path = 'csv/00-cleaned-up-data.csv'
df = pd.read_csv(data_path)
print('number of observations in dataset: ', df.shape)

# ## training data:
# x_train = pd.read_csv('csv/02-x_train.csv')
# y_train = pd.read_csv('csv/02-y_train.csv')

# ## test data:
# x_test = pd.read_csv('csv/02-x_test.csv')
# y_test = pd.read_csv('csv/02-y_test.csv')

### CORRELATION: ------------------------------

#--- engine location value count shows only 1 category; so it is dropped
# print(df['engine-location'].value_counts())
df = df.drop(columns=['engine-location'])

#--- correlation plots and regression line equations:

for key in df.keys():

    if key != 'price' and df[key].dtype != 'O':
        
        print(key)
        # print(ohe[key].shape, ohe['price'].shape)
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(df[key], df['price'])
        
        # save plot for visual inspection
        fig = plt.figure()
        sns.regplot(x=key, y="price", data=df)
        plt.title('Slope: ' + str(slope) + '; Intercept: ' + str(intercept))
        plt.savefig('plots/feature-influence/06-a-reg-plot-'+ key +'.png')
        plt.close(fig)

        print('saved plot for: ', key, '; reg-line slope: ', slope, '; slope-angle: ', np.rad2deg(np.arctan(slope)))

        # from visual inspection of correlation plots, any regression line with an abolsute value of 5000 is assumed to indicate a strong correlation between predictor and target; only those will be used to train the neural net; the rest of the predictor variables will be dropped

        # if abs(slope) < 4000:
        #     df = df.drop(columns=[key])
            
        # print(key, slope)

### NARROW DOWN FEATURES: ======================================================
# this one-hot encoding makes a total of 69 predictor variables to predict the target (price) variable
# the dataset contains 159 observations, and 69 features (after one-hot-ecoding) leads to NaN outputs
# the output mostly result in NaNs

# so only a few features will be used at a time to train models; 
# the correlation charts are consulted, and the predictors with strong correlation are chosen

## group 1: 
group1 = np.array(['engine-size','horsepower','city-mpg','highway-mpg','body-style','price'])

df = df[group1]
# print(df.keys())

### PRE-PROCESSING: ===========================================================

## check for NaN: -------------------------------------------
#print(df.isna().sum())

## one-hot all categorical variables:  --------------------------------------- 

#--- list all predictor variable names:
# print(df.columns)

# Index(['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
#        'num-of-doors', 'body-style', 'drive-wheels',
#        'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
#        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
#        'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
#        'highway-mpg', 'price'],
#       dtype='object')

#--- isolate categorical variables:
# symboling
# make
# fuel-type
# aspiration
# num-of-doorsS
# body-style
# drive-wheels
# engine-type
# number-of-cylinders
# fuel-system

# apply 'one-hot-encoding": --------------------------------------------------

ohe = pd.get_dummies(df)
#ohe = pd.get_dummies(ohe, columns=['symboling'])
# print(ohe.keys())

### TEST-TRAIN SPLIT: ==========================================================

train_data = ohe.sample(frac=0.8,random_state=0)
test_data = ohe.drop(train_data.index)

### CONSOLE LOG DATA: ==========================================================

# print(train_data.columns)
# print(train_data.tail())
# print(train_data.isna().sum())

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
        keras.layers.Dense(128, activation = 'sigmoid', kernel_initializer=keras.initializers.RandomNormal(mean=0.001, stddev=0.05, seed=1), input_dim = len(normed_train_data.keys())),
        keras.layers.Dense(33, activation = 'sigmoid'),
        #keras.layers.Dense(11, activation = 'relu'),
        keras.layers.Dense(1, activation = 'relu')
    ])

    model.compile(
                    loss = 'mse',
                    #optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.001, nesterov=True),
                    optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
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
