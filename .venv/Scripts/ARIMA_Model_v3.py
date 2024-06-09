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
# Filter out prices that are negative or zero
df = df[df[value_column_name] > 0]

# Convert the date column to datetime
df['Date'] = pd.to_datetime(df['Date'], utc=True)

# Convert the data into logarithmic output to stabilize the variance
df[value_column_name] = np.log(df[value_column_name])

# Split the data into training and test sets
train_idx, test_idx = train_test_split(df.index, test_size=0.2, shuffle=False)
train_data = df.loc[train_idx][value_column_name]
test_data = df.loc[test_idx][value_column_name]

# Check for stationarity of the training data
result_train = adfuller(train_data)
stationary = result_train[1] <= 0.05

# Fit the ARIMA model
train_data.reset_index(drop=True, inplace=True)
model_arima = ARIMA(train_data, order=(1, 1, 1))
model_arima_fit = model_arima.fit()
print(model_arima_fit.summary())

# Evaluate ARIMA model performance
arima_forecast = model_arima_fit.forecast(steps=len(test_data))
arima_mae = mean_absolute_error(test_data, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))

# Fit the SARIMAX model
model_sarimax = SARIMAX(train_data, order=(1, 1, 1))
model_sarimax_fit = model_sarimax.fit(disp=False)

# Evaluate SARIMAX model performance
sarimax_forecast = model_sarimax_fit.forecast(steps=len(test_data))
sarimax_mae = mean_absolute_error(test_data, sarimax_forecast)
sarimax_rmse = np.sqrt(mean_squared_error(test_data, sarimax_forecast))

# Forecast for future periods
forecast_steps = pd.date_range(start='2023-12-31', end='2024-04-01', freq='D')
sarimax_future_forecast = model_sarimax_fit.get_forecast(steps=len(forecast_steps))
sarimax_future_forecast_values = sarimax_future_forecast.predicted_mean

# Convert log-transformed predictions back to original scale
sarimax_future_forecast_values = np.exp(sarimax_future_forecast_values)

# Model performance evaluation
mae = mean_absolute_error(test_data, sarimax_forecast)
rmse = np.sqrt(mean_squared_error(test_data, sarimax_forecast))

# Plot the last three years with actual and forecasted values
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][-3*365:], np.exp(df[value_column_name][-3*365:]), label='Actual Prices')
plt.plot(forecast_steps, sarimax_future_forecast_values, label='Forecasted Prices', color='red')
plt.title('Day Ahead Prices: Actual vs Forecasted')
plt.xlabel('Date')
plt.ylabel(value_column_name)
plt.legend()
plt.grid(True)
plt.show()

# Print performance metrics
print("ARIMA Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): {arima_mae}")
print(f"Root Mean Squared Error (RMSE): {arima_rmse}\n")

print("SARIMAX Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): {sarimax_mae}")
print(f"Root Mean Squared Error (RMSE): {sarimax_rmse}")

performance_metrics = {
    "SARIMAX MAE": sarimax_mae,
    "SARIMAX RMSE": sarimax_rmse,
    "ARIMA MAE": arima_mae,
    "ARIMA RMSE": arima_rmse
}

performance_metrics_df = pd.DataFrame(performance_metrics, index=[0])
performance_metrics_df

# Evaluation
print(model_sarimax_fit.summary())




n_periods = 3
true_values = test_data[:n_periods]

# Ensuring both data sets have the same length
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

