import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import itertools
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMAResults

# Define directory and file path
if os.getenv('PYTHON_ENV') == 'pycharm':
    data_dir = 'C:/Users/MichalZlotnik/PycharmProjects/ElectricityGermany/data'
else:
    data_dir = '/content/ElectricityGermany/data'

file_name = 'day_ahead_price_germany_0.csv'
file_path = os.path.join(data_dir, file_name)

# Read the data
df = pd.read_csv(file_path)

# Ensure the 'date' column is the index and convert it to datetime
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Ensure the 'value' column is of type float
df['value'] = df['value'].astype(float)

# Print data types to confirm
print(df.dtypes)
print(df)

# Plot the data
df.plot(figsize=(15, 6))
plt.show()

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, d, and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, d, and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# Initialize DataFrame and array for storing results
df_arimas_aic = pd.DataFrame(columns=['param', 'AIC'])
aic_values = np.array([])

# Specify to ignore warning messages
warnings.filterwarnings("ignore")
'''
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df['value'],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            string_row = 'ARIMA{}x{}12'.format(param, param_seasonal)
            string_row_aic = 'ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic)
            print(string_row_aic)

            # Append to DataFrame
            new_row = pd.DataFrame({'param': [string_row], 'AIC': [results.aic]})
            df_arimas_aic = pd.concat([df_arimas_aic, new_row], ignore_index=True)

            # Append to NumPy array
            aic_values = np.append(aic_values, results.aic)
        except Exception as e:
            print(f"Error with parameters {param} {param_seasonal}: {e}")
            continue

print("DataFrame with AIC values:")
print(df_arimas_aic)
print("\nNumPy array with AIC values:")
print(aic_values)
'''

mod = sm.tsa.statespace.SARIMAX(df,
                                order=(1, 1, 1),
                                seasonal_order=(1, 0, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2024-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = df['2023-01-01':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Price')
plt.legend()

plt.show()


y_forecasted = pred.predicted_mean
y_truth = df['2024-01-01':]['value']

# Compute the mean square error
#mse = mean_squared_error(y_truth, y_forecasted)
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_truth, y_forecasted)
print(f'Mean Absolute Error (MAE): {mae}')

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_truth - y_forecasted) / y_forecasted)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape}')

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_truth, y_forecasted))
print(f'Root Mean Squared Error (RMSE): {rmse}')


pred_dynamic = results.get_prediction(start=pd.to_datetime('2024-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()


ax = df['2023-01-01':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2024-01-01'), df.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Date')
ax.set_ylabel('Price')

plt.legend()
plt.show()

# Extract the predicted and true values of our time series
y_forecasted = pred_dynamic.predicted_mean
y_truth = df['2024-01-01':]['value']

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


# Get forecast 30 steps ahead in future
pred_uc = results.get_forecast(steps=30)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

ax = df.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Price')

plt.legend()
plt.show()


# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_truth, y_forecasted)
print(f'Mean Absolute Error (MAE): {mae}')

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_truth - y_forecasted) / y_forecasted)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape}')

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_truth, y_forecasted))
print(f'Root Mean Squared Error (RMSE): {rmse}')