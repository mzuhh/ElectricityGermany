import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Importing the day ahead prices
data_dir = 'C:/Users/MichalZlotnik/PycharmProjects/ElectricityGermany/data'
file_name = 'day_ahead_price_germany.csv'
file_path = os.path.join(data_dir, file_name)
df = pd.read_csv(file_path)

# Filter out prices that are negative or zero
df = df[df['Day Ahead Auction Price (EUR/MWh)'] > 0]

# Converting the date column to datetime
df['Date (GMT+1)'] = pd.to_datetime(df['Date (GMT+1)'], utc=True)

# Getting basic information about the data
df.info()

# Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(df['Date (GMT+1)'], df['Day Ahead Auction Price (EUR/MWh)'])
plt.title('Day Ahead Prices')
plt.xlabel('Date (GMT+1)')
plt.ylabel('Day Ahead Auction Price (EUR/MWh)')
plt.grid(True)
plt.show()

# Converting the data into logarithmic output to stabilize the variance
df['Day Ahead Auction Price (EUR/MWh)'] = np.log(df['Day Ahead Auction Price (EUR/MWh)'])

# Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(df['Date (GMT+1)'], df['Day Ahead Auction Price (EUR/MWh)'])
plt.title('Day Ahead Prices (Log Transformed)')
plt.xlabel('Date (GMT+1)')
plt.ylabel('Log Day Ahead Auction Price (EUR/MWh)')
plt.grid(True)
plt.show()

# Splitting the data into test and training set
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
print("Train set size: ", len(train_idx))
print("Test set size", len(test_idx))

# Check for stationarity of time series which is a requirement of ARIMA Models
train_data = df.loc[train_idx]['Day Ahead Auction Price (EUR/MWh)']
test_data = df.loc[test_idx]['Day Ahead Auction Price (EUR/MWh)']

result_train = adfuller(train_data)

# Printing the ADF test results for the training set
print('ADF Statistic (Training Set):', result_train[0])
print('p-value (Training Set):', result_train[1])

if result_train[1] <= 0.05:
    print('Reject the null hypothesis: Data is stationary')
else:
    print('Fail to reject the null hypothesis: Data is non-stationary')

# Fitting the ARIMA Model
model = ARIMA(train_data, order=(3, 0, 2))
model_fit = model.fit()

# Print the summary of the ARIMA model
print(model_fit.summary())

# Auto ARIMA fit using train_data directly as a series
auto_model = auto_arima(train_data, maxiter=100, seasonal=False, stepwise=True, suppress_warnings=True)
print("AUTO ARIMA RESULTS")
print(auto_model.summary())

# Forecasting Electricity Prices
n_periods = 10  # Number of future periods to forecast

# Generate future timestamps for the forecasted periods
future_dates = pd.date_range(start='2024-04-10', periods=n_periods, freq='H')

# Make predictions for the future time periods
forecast = model_fit.get_forecast(steps=n_periods)

# Extract the forecasted values and confidence intervals
forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Print the forecasted values and confidence intervals for the future time periods
print(f"Forecasted Values:\n{forecast_values}")
print(f"Confidence Intervals:\n{confidence_intervals}")

# Optionally, plot the forecasted values and confidence intervals
plt.figure(figsize=(10, 5))
plt.plot(future_dates, forecast_values, label='Forecasted Values')
plt.fill_between(future_dates, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='gray', alpha=0.2, label='Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Forecasted Value')
plt.title('ARIMA Forecast')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation
true_values = test_data

# Ensuring both data sets have the same length
min_len = min(len(true_values), len(forecast_values))
true_values = true_values[:min_len]
forecast_values = forecast_values[:min_len]

# Transform the data back from log scale
true_values_exp = np.exp(true_values)
forecast_values_exp = np.exp(forecast_values)

# Ensure there are no zero or negative values in true_values_exp to avoid division by zero in MAPE calculation
true_values_exp = true_values_exp.replace(0, np.nan).dropna()

# Align forecast values length with true values after dropping NaNs
forecast_values_exp = forecast_values_exp[:len(true_values_exp)]

# Mean Absolute Error (MAE)
mae = mean_absolute_error(true_values_exp, forecast_values_exp)

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((true_values_exp*1000 - forecast_values_exp*1000) / true_values_exp*1000)) * 100

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(true_values_exp, forecast_values_exp))

print(min(true_values_exp))
print(min(forecast_values_exp))

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
print(f'Root Mean Squared Error (RMSE): {rmse}')