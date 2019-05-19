# import ml libraries 
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

### IMPORT: ------------------------------------
data_path = 'csv/00-cleaned-up-data.csv'
df = pd.read_csv(data_path)

### ASSIGN PREDICTOR/TARGET: -------------------
y = df['price']
x = df.loc[:, df.columns != 'price']
print(y.shape, x.shape)

### TEST-TRAIN SPLIT: --------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:", x_train.shape[0])
print(x_train.shape,y_train.shape)

### WRITE TO FILE: -----------------------------
df.to_csv('csv/02-x_train.csv', index=False)
df.to_csv('csv/02-y_train.csv', index=False)
df.to_csv('csv/02-x_test.csv', index=False)
df.to_csv('csv/02-y_test.csv', index=False)