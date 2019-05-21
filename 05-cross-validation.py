# import ml libraries 
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

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

#### K-FOLDS CROSS-VALIDATION: ===============================================

## Average R-squared Value: 

# 2-folds:
print('\n')
print('2-Folds train-test splits - R^2 error: ')
print(cross_val_score(lm_engine_size, x_train[['engine-size']], y_train, cv=2))
print('\n')
print('2-Folds - R^2 average: ')
print(np.mean(cross_val_score(lm_engine_size, x_train[['engine-size']], y_train, cv=2)))
print('\n') 

# 3-folds:
print('3-Folds train-test splits - R^2 error: ')
print(cross_val_score(lm_engine_size, x_train[['engine-size']], y_train, cv=3))
print('\n')
print('3-Folds - R^2 average: ')
print(np.mean(cross_val_score(lm_engine_size, x_train[['engine-size']], y_train, cv=3)))
print('\n')

# 4-folds:
print('4-Folds train-test splits - R^2 error: ')
print(cross_val_score(lm_engine_size, x_train[['engine-size']], y_train, cv=4))
print('\n')
print('4-Folds - R^2 average: ')
print(np.mean(cross_val_score(lm_engine_size, x_train[['engine-size']], y_train, cv=4)))
print('\n')

# 5-folds:
print('5-Folds train-test splits - R^2 error: ')
print(cross_val_score(lm_engine_size, x_train[['engine-size']], y_train, cv=5))
print('\n')
print('5-Folds - R^2 average: ')
print(np.mean(cross_val_score(lm_engine_size, x_train[['engine-size']], y_train, cv=5)))
print('\n')

#### K-FOLDS PREDICTION: ===============================================

## c) out-of-sample evaluation --------------------------------------

# Distribution Plots --------------------------------------

plt.figure(0, figsize=(12, 10))

ax1 = sns.distplot(y_test, hist=False, color="r", label='True Values')
ax2 = sns.distplot(cross_val_predict(lm_engine_size, x_test[['engine-size']], y_test, cv=4), hist=False, color="b", label='Predicted Values', ax = ax1)

plt.title('Engine-Size:Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.savefig('plots/05-a-dist-plot-engine-size.png')

