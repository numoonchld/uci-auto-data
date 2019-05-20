# import ml libraries 
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

### IMPORT: ------------------------------------
x_train = pd.read_csv('csv/02-x_train.csv')
y_train = pd.read_csv('csv/02-y_train.csv')

### INITIALIZE LINEAR REGRESSION ALGORITHM BASED MODEL: ----------
lm_engine_size = LinearRegression()
lm_horsepower = LinearRegression()
lm_highway_mpg = LinearRegression()
lm_symboling = LinearRegression()
lm_norm_loss = LinearRegression()

lm_build_dim = LinearRegression()
lm_car_specs = LinearRegression()

### FIT DATA TO MODEL: -------------------------------

## simple linear regression
lm_engine_size.fit(x_train[['engine-size']], y_train)
lm_horsepower.fit(x_train[['horsepower']], y_train)
lm_highway_mpg.fit(x_train[['highway-mpg']], y_train)
lm_symboling.fit(x_train[['symboling']], y_train) 
lm_norm_loss.fit(x_train[['normalized-losses']], y_train)

## multiple linear regression
lm_build_dim.fit(x_train[['length','width','height','curb-weight']], y_train) 
lm_car_specs.fit(x_train[['engine-size','horsepower','city-mpg','highway-mpg']], y_train)

### INSPECT MODEL EQUATIONS: ---------------------
print('Engine-Size:Price Linear Regression Model: ')
print(lm_engine_size.intercept_, lm_engine_size.coef_)
print('---------')

print('Horsepower:Price Linear Regression Model: ')
print(lm_horsepower.intercept_, lm_horsepower.coef_)
print('---------')

print('Highway-MPG:Price Linear Regression Model: ')
print(lm_highway_mpg.intercept_, lm_highway_mpg.coef_)
print('---------')

print('Symboling:Price Linear Regression Model: ')
print(lm_symboling.intercept_, lm_symboling.coef_)
print('---------')

print('Normalized-Losses:Price Linear Regression Model: ')
print(lm_norm_loss.intercept_, lm_norm_loss.coef_)
print('---------')

print('Build Dimensions:Price Linear Regression Model: ')
print(lm_build_dim.intercept_, lm_build_dim.coef_)
print('---------')

print('Car Specs:Price Linear Regression Model: ')
print(lm_car_specs.intercept_, lm_car_specs.coef_)
print('---------')


### PREDICTIONS: -------------------------------



