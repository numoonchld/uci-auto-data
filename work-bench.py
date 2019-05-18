# import ml libraries 
import pandas as pd
import numpy as np
import scipy as sp

import matplotlib as mlp
import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf
import keras as kr

##------------------------------------

# import data
data_path = 'imports-85.data'
df = pd.read_csv(data_path)

# create header 
header = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']

# add header
df.columns = header
print(df.head(5))
print(df.tail(5))

# check datatypes of each column
# print(df.dtypes)

# quick stats - some columns
# print(df.describe())

# quick stats - all columns
# print(df.describe(include="all"))

# print top 30 and bottom 30 rows 
# print(df.info)

# lists different types of classificaitons under each column
print(df['body-style'].value_counts())



## clean-up N/A in data---------------------------------------------------

# 8. Missing Attribute Values: (denoted by "?")
#    Attribute #:   Number of instances missing a value:
#    2.             41
#    6.             2
#    19.            4
#    20.            4
#    22.            2
#    23.            2
#    26.            4

df.dropna()  # drops entire observation with missing values


