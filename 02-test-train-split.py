# import ml libraries 
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

### IMPORT: ------------------------------------
data_path = 'csv/00-cleaned-up-data.csv'
df = pd.read_csv(data_path)

### MAKE IT ML-ALGORITHM FRIENDLY: -----------------------

## convert categorical predictor variables to quantitative variables 
df = pd.get_dummies(df)
                             

### ASSIGN PREDICTOR/TARGET: -------------------
y = df['price']
x = df.loc[:, df.columns != 'price']
print(y.shape, x.shape)

### TEST-TRAIN SPLIT: --------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:", x_train.shape[0])
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

### WRITE TO FILE: -----------------------------
x_train.to_csv('csv/02-x_train.csv', index=False)
y_train.to_frame('price').to_csv('csv/02-y_train.csv', index=False)
x_test.to_csv('csv/02-x_test.csv', index=False)
y_test.to_frame('price').to_csv('csv/02-y_test.csv', index=False)
