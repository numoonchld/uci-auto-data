# import ml libraries 
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pickle

### IMPORTS: ==================================================================

## unsplit data:
data_path = 'csv/00-cleaned-up-data.csv'
df = pd.read_csv(data_path)

## training data:
x_train = pd.read_csv('csv/02-x_train.csv')
y_train = pd.read_csv('csv/02-y_train.csv')

## test data:
x_test = pd.read_csv('csv/02-x_test.csv')
y_test = pd.read_csv('csv/02-y_test.csv')

## trained models: 

# simple linear regression
lm_engine_size = pickle.load(open('models/lm_engine_size.sav', 'rb'))
lm_horsepower = pickle.load(open('models/lm_horsepower.sav', 'rb'))
lm_highway_mpg = pickle.load(open('models/lm_highway_mpg.sav', 'rb'))
lm_symboling = pickle.load(open('models/lm_symboling.sav', 'rb'))
lm_norm_loss = pickle.load(open('models/lm_norm_loss.sav', 'rb'))

# multiple linear regression
lm_build_dim = pickle.load(open('models/lm_build_dim.sav', 'rb'))
lm_car_specs = pickle.load(open('models/lm_car_specs.sav', 'rb'))

### EVALUATION: =======================================================

## in-sample evaluation --------------------------------------

## VISUAL

## a) Residual Plots --------------------------------------

# simple linear regression
plt.figure(0, figsize=(12, 10))
sns.residplot(lm_horsepower.predict(x_train[['engine-size']]), y_train)
plt.savefig('plots/04-a-resid-plot-engine-size.png')

plt.figure(1, figsize=(12, 10))
sns.residplot(lm_horsepower.predict(x_train[['horsepower']]), y_train)
plt.savefig('plots/04-a-resid-plot-horsepower.png')

plt.figure(2, figsize=(12, 10))
sns.residplot(lm_highway_mpg.predict(x_train[['highway-mpg']]), y_train)
plt.savefig('plots/04-a-resid-plot-highway_mpg.png')

plt.figure(3, figsize=(12, 10))
sns.residplot(lm_norm_loss.predict(x_train[['normalized-losses']]), y_train)
plt.savefig('plots/04-a-resid-plot-norm-loss.png')

# multiple linear regression
plt.figure(4, figsize=(12, 10))
sns.residplot(lm_build_dim.predict(x_train[['length', 'width', 'height', 'curb-weight']]), y_train)
plt.savefig('plots/04-a-resid-plot-build-dim.png')

plt.figure(5, figsize=(12, 10))
sns.residplot(lm_car_specs.predict(x_train[['engine-size', 'horsepower', 'city-mpg', 'highway-mpg']]), y_train)
plt.savefig('plots/04-a-resid-plot-car-specs.png')

## b) Distribution Plots --------------------------------------

plt.figure(6, figsize=(12, 10))

ax1 = sns.distplot(y_train, hist=False, color="r", label='True Values')
ax2 = sns.distplot(lm_engine_size.predict(x_train[['engine-size']]), hist=False, color="b", label='Predicted Values', ax=ax1)

plt.title('Engine-Size:Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.savefig('plots/04-b-dist-plot-engine-size.png')

# plt.show()

# R^2 --------------------------------------
print('\n')
print('in-sample R^2 values for trained models: ------------------')
print('\n')

# simple linear regression
print('engine-size:price') 
print(lm_engine_size.score(x_train[['engine-size']], y_train))
print('\n')


# MSE --------------------------------------
print('\n')
print('in-sample MSE values for trained models: ------------------')
print('\n')

# simple linear regression
print('engine-size:price') 
print(mean_squared_error(y_train,lm_engine_size.predict(x_train[['engine-size']])))
print('\n')


## c) out-of-sample evaluation --------------------------------------

# Distribution Plots --------------------------------------

plt.figure(7, figsize=(12, 10))

ax1 = sns.distplot(y_test, hist=False, color="r", label='True Values')
ax2 = sns.distplot(lm_engine_size.predict(x_test[['engine-size']]), hist=False, color="b", label='Predicted Values', ax=ax1)

plt.title('Engine-Size:Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.savefig('plots/04-c-dist-plot-engine-size.png')

# R^2 --------------------------------------
print('\n')
print('out-of-sample R^2 values for trained models: ------------------')
print('\n')

# simple linear regression
print('engine-size:price') 
print(lm_engine_size.score(x_test[['engine-size']], y_test))
print('\n')

# MSE --------------------------------------
print('\n')
print('out-of-sample MSE values for trained models: ------------------')
print('\n')

# simple linear regression
print('engine-size:price') 
print(mean_squared_error(y_test,lm_engine_size.predict(x_test[['engine-size']])))
print('\n')

### GENERALIZATION ERROR: =======================================================

