
# import ml libraries 
import pandas as pd
import numpy as np


### IMPORT: ------------------------------------
data_path = 'imports-85.data'
df = pd.read_csv(data_path)

### INSPECT: -----------------------------------

# inspect data, print top 30 and bottom 30 rows:
# print(df.head(5))
# print(df.tail(5))
# print(df.info)

# lists different types of classificaitons under each column:
# print(df['body-style'].value_counts())

### CLEANUP: -------------------------------------

## add columns headers:
header = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']

df.columns = header

## missing data cleanup:

# 8. Missing Attribute Values: (denoted by "?")
#    Attribute #:   Number of instances missing a value:
#    2.             41
#    6.             2
#    19.            4
#    20.            4
#    22.            2
#    23.            2
#    26.            4

# convert "?" to np.nan:
df.replace('?',np.nan, inplace=True)
df.dropna(inplace=True)  # drops entire observation with missing values

# additional tools:
# df.dropna(subset=['price'],axis = 0,inplace=True) # inplace = true drops the rows 
# df = df.dropna(subset=['price'],axis = 0)

## datatype cleanup: 

# check datatypes of each column:
#print(df.head(30))
#print(df.dtypes)

# apply data type conversion fixes:
df['normalized-losses'] = df['normalized-losses'].astype('int')
df['bore'] = df['bore'].astype('float')
df['stroke'] = df['stroke'].astype('float')
df['horsepower'] = df['horsepower'].astype('float')
df['peak-rpm'] = df['peak-rpm'].astype('int')
df['price'] = df['price'].astype('int')

# recheck data type correction:
# print(df.dtypes)

### PRELIM STATS: --------------------------------------

# quick stats - some columns:
# print(df.describe())

# quick stats - all columns:
print(df.describe(include="all"))

### EXPORT CSV: -----------------------------------------

df.to_csv('csv/00-cleaned-up-data.csv', index=False)