import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


date_column_name = 'Date'

# Dynamic Directories
if os.getenv('PYTHON_ENV') == 'pycharm':

    data_dir = 'C:/Users/MichalZlotnik/PycharmProjects/ElectricityGermany/data'
else:
    data_dir = '/content/ElectricityGermany/data'

file_name = 'day_ahead_price_germany.csv'
file_path = os.path.join(data_dir, file_name)
print(f"Loading data from {file_path}")



df = pd.read_csv(file_path)
df1 = df
print(df.head(10))

df1[date_column_name] = pd.to_datetime(df1[date_column_name], format='%Y-%m-%d', utc=True)
#daily
davg_df1 = df1.groupby(pd.Grouper(freq='D', key=date_column_name)).mean()
print(davg_df1.head(10))



# Convert the date column to datetime
df[date_column_name] = pd.to_datetime(df[date_column_name], utc=True)

# Filter out prices that are negative or zero
df = df[df[value_column_name] > 0]
print(df.head(10))

# Convert the data into logarithmic output to stabilize the variance
df[value_column_name] = np.log(df[value_column_name])
print(df.head(10))


