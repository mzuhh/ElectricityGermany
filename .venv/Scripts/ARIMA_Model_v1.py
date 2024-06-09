import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMAResults

date_column_name = 'date'
value_column_name = 'value'

#Dynamic Directories
if os.getenv('PYTHON_ENV') == 'pycharm':
    data_dir = 'C:/Users/MichalZlotnik/PycharmProjects/ElectricityGermany/data'
else:
    data_dir = '/content/ElectricityGermany/data'

file_name = 'day_ahead_price_germany_0.csv'
file_path = os.path.join(data_dir, file_name)
print(f"Loading data from {file_path}")




df = pd.read_csv(file_path)

# Filterinng out prices that are negative or zero

df = df[df[value_column_name] > 0]

# Converting the date column to account for the time stamps in it
df[date_column_name] = pd.to_datetime(df[date_column_name], utc=True)


# Getting basic information about the data
df.info()

# Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(df[date_column_name], df[value_column_name])
plt.title('Day Ahead Prices')
plt.xlabel(date_column_name)
plt.ylabel(value_column_name)
plt.grid(True)
plt.show()

# Converting the data into logarithmic output to stabilize the variance
df[value_column_name] = np.log(df[value_column_name]) # transform the data back when making real predictions


#Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(df[date_column_name], df[value_column_name])
plt.title('Day Ahead Prices')
plt.xlabel(date_column_name)
plt.ylabel(value_column_name)
plt.grid(True)
plt.show()

# Splitting the data into test and training set

X = df[[date_column_name, value_column_name]]

train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
print("Train set size: ", len(train_idx))
print("Test set size", len(test_idx))

# Check for stationarity of time series which is a requirement of ARIMA Models


data = df[value_column_name]

train_data = df.loc[train_idx][value_column_name]
test_data = df.loc[test_idx][value_column_name]

result_train = adfuller(train_data)

# Printing the ADF test results for both training and test set
print('ADF Statistic (Training Set):', result_train[0])
print('p-value (Training Set):', result_train[1])

if result_train[1] <= 0.05:
    print('Reject the null hypothesis: Data is stationary')
else:
    print('Fail to reject the null hypothesis: Data is non-stationary')

# Fitting the ARIMA Model
from statsmodels.tsa.arima.model import ARIMA

# Reset the index of the Series
train_data.reset_index(drop=True, inplace=True)

 # Fit the ARIMA model using train_data directly as a Series
model = ARIMA(train_data, order=(1, 1, 1))
model_fit = model.fit()

# Print the summary of the ARIMA model
print(model_fit.summary())

#Forecasting Electricity Prices
n_periods = 1  # Number of future periods to forecast

# Generate future timestamps for the forecasted periods
future_dates = pd.date_range(start='2023-04-18', periods=n_periods, freq='D')

# Make predictions for the future time periods
forecast = model_fit.get_forecast(steps=n_periods)

# Extract the forecasted values and confidence intervals
forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Print the forecasted values and confidence intervals for the future time periods
print(f"Forecasted Values:\n{forecast_values}")
print(f"Confidence Intervals:\n{confidence_intervals}")

# Optionally, plot the forecasted values and confidence intervals
forecast_values.plot(label='Forecasted Values')
plt.fill_between(future_dates, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='gray', alpha=0.2, label='Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Forecasted Value')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()

#Evaluation


true_values = test_data

#Ensuring both data sets have the same length
min_len = min(len(true_values), len(forecast_values))
true_values = true_values[:min_len]
forecast_values = forecast_values[:min_len]

# Mean Absolute Error (MAE)
mae = mean_absolute_error(true_values, forecast_values)
print(f'Mean Absolute Error (MAE): {mae}')

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((true_values - forecast_values) / true_values)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape}')

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(true_values, forecast_values))
print(f'Root Mean Squared Error (RMSE): {rmse}')


print(f'\nEOT')