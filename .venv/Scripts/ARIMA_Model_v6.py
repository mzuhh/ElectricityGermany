import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import boxcox
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima.arima import auto_arima

# Define directory and file path
if os.getenv('PYTHON_ENV') == 'pycharm':
    data_dir = 'C:/Users/MichalZlotnik/PycharmProjects/ElectricityGermany/data'
else:
    data_dir = '/content/ElectricityGermany/data'

file_name = 'day_ahead_price_germany_0.csv'
file_path = os.path.join(data_dir, file_name)
df = pd.read_csv(file_path)


date_column_name = 'date'
value_column_name = 'value'
start_date_training = "2015-01-01"
end_date_training = "2022-01-01"
start_date_testing = "2022-01-01"
plot_main_title  = "Daily Price"
nsteps = 40



df[date_column_name] = pd.to_datetime(df[date_column_name], format="%Y-%m-%d", errors="coerce")
df = df[df[value_column_name] > 0]
# df = df.set_index(date_column_name)[value_column_name]

print(df.head())

df_daily_training = df[start_date_training:end_date_training]
df_daily_test = df[start_date_testing:]





plt.figure(figsize=(18, 6))
plt.title(plot_main_title , fontsize=14)
df_daily_training.plot(label="training set"+start_date_training+"-"+end_date_training, fontsize=14)
df_daily_test.plot(label="test set "+start_date_testing, fontsize=14)

plt.legend()
plt.show()

# Box-Cox transformation
from scipy.stats import boxcox

boxcox_transformed_data, boxcox_lamba = boxcox(df_daily_training + 10)
boxcox_transformed_data = pd.Series(
    boxcox_transformed_data, index=df_daily_training.index
)



fig, ax = plt.subplots(2, 1, figsize=(16, 8))
df_daily_training.plot(ax=ax[0], color="black", fontsize=14)
ax[0].set_title("Original time series", fontsize=14)


boxcox_transformed_data.plot(
    ax=ax[1],
    color="grey",
)
ax[1].set_title("Box-Cox transformed time series", fontsize=14)

ax[0].grid()
ax[1].grid()

plt.tight_layout()
plt.show()

#Stationary check
# KPSS test
from statsmodels.tsa.stattools import kpss


def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f"KPSS Statistic: {statistic}")
    print(f"p-value: {p_value}")
    print(f"num lags: {n_lags}")
    print("Critial Values:")
    for key, value in critical_values.items():
        print(f"   {key} : {value}")
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')

kpss_test(df_daily_training)

df_daily_training_diff1 = df_daily_training.diff()

kpss_test(df_daily_training_diff1.dropna())  ## ignore NaN for kpss

plt.figure(figsize=(18, 6))
plt.title("Differenced data set: df_daily_training_diff1`")
df_daily_training_diff1.plot(color="black")

plt.grid()
plt.show()




from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

plt.figure(figsize=(18, 6))


plot_acf(df_daily_training_diff1.dropna())
plt.title("ACF for Differenced Series")
#The plot shows significant spikes at a lag of 1, 7, 14, 21, 28, 35 years.

plt.show()



fig, ax = plt.subplots(1, 2, figsize=(13, 5))

plot_acf(df_daily_training_diff1.dropna(), ax=ax[0])
ax[0].set_title("ACF")


plot_pacf(
    df_daily_training_diff1.dropna(), method="ywm", ax=ax[1]
)  ## add the calculation method running in the background ("ywm")

ax[1].set_title("PACF")
plt.show()

#ARIMA(1,1,3)

#Model Selection



from statsmodels.tsa.arima.model import ARIMA

'''
model = ARIMA(df_daily_training, order=(1, 1, 3))
model_fit = model.fit()
print(f"ARIMA(1, 1, 3) - AICc: {round(model_fit.aicc,2)}")

model = ARIMA(df_daily_training, order=(1, 1, 2))
model_fit = model.fit()
print(f"ARIMA(1, 1, 2) - AICc: {round(model_fit.aicc,2)}")

print(f"ARIMA(0, 1, 2) - AICc: {round(model_fit.aicc,2)}")

model = ARIMA(df_daily_training, order=(1, 1, 1))
model_fit = model.fit()
print(f"ARIMA(1, 1, 1) - AICc: {round(model_fit.aicc,2)}")
'''



# fit model
model = ARIMA(df_daily_training, order=(5, 1, 3))
model_fit = model.fit()
print(model_fit.summary())



#Auto arima

from pmdarima.arima import auto_arima

auto_model = auto_arima(df_daily_training)
print(auto_model.summary())


#Check the residuals
residuals = pd.DataFrame(auto_model.resid())


fig, ax = plt.subplots(1, 2, figsize=(13, 5))

residuals.plot(ax=ax[0], legend=False)
ax[0].grid()
ax[0].set_title("Residuals of ARIMA(1, 1, 3))")
plot_acf(residuals, ax=ax[1])
ax[1].set_title("ACF of Residuals")
plt.show()


#Ljung-Box test
from statsmodels.stats.diagnostic import acorr_ljungbox
# perform Ljung-Box test on residuals with lag=10
Btest = acorr_ljungbox(
    auto_model.resid(), lags=[10], return_df=True, model_df=5
)  ## ARIMA model_df = lags - df , df =  p +q

print(Btest)



#Calculate forecasts
model = ARIMA(df_daily_training, order=(5, 1, 3))
fitted = model.fit()
forecast_series = fitted.forecast(steps=nsteps, alpha=0.05)
plt.figure(figsize=(13, 4))
plt.title(plot_main_title)
df_daily_training.plot(color="black", label="training set" + start_date_training + " " + end_date_training)
plt.plot(fitted.fittedvalues, color="blue", label="fitted values")
plt.legend()
plt.show()



# Perform rolling forecast
history = list(df_daily_training)
predictions = []
conf_ints_95 = []
conf_ints_80 = []




for t in range(len(df_daily_test)):
    model = ARIMA(history, order=(5, 1, 3))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=1)
    yhat = forecast.predicted_mean[0]
    conf_int_95 = forecast.conf_int(alpha=0.05)
    conf_int_80 = forecast.conf_int(alpha=0.2)
    predictions.append(yhat)
    conf_ints_95.append(conf_int_95[0])
    conf_ints_80.append(conf_int_80[0])
    history.append(df_daily_test[t])

# Convert predictions and confidence intervals to series
forecast_index = df_daily_test.index
predictions_series = pd.Series(predictions, index=forecast_index)
conf_ints_95_df = pd.DataFrame(conf_ints_95, index=forecast_index, columns=['lower value', 'upper value'])
conf_ints_80_df = pd.DataFrame(conf_ints_80, index=forecast_index, columns=['lower value', 'upper value'])



# Plot the results
start_date_plot = "2022-01-01"


plt.figure(figsize=(13, 4))
plt.title(plot_main_title)
df_daily_training[start_date_plot:].plot(color="black", label="training set" + start_date_training + " " + end_date_training)
df_daily_test.plot(color="red", label="test set "+ start_date_testing)
predictions_series.plot(label="forecast", color="blue")

# Plot confidence intervals
plt.fill_between(
    conf_ints_95_df.index,
    conf_ints_95_df["lower value"],
    conf_ints_95_df["upper value"],
    color="b",
    alpha=0.1,
)

plt.fill_between(
    conf_ints_80_df.index,
    conf_ints_80_df["lower value"],
    conf_ints_80_df["upper value"],
    color="b",
    alpha=0.2,
)

plt.legend()
plt.show()

# Calculate error metrics
true_values = df_daily_test.values
forecast_values = predictions_series.values

# Mean Absolute Error (MAE)
mae = mean_absolute_error(true_values, forecast_values)
print(f'Mean Absolute Error (MAE): {mae}')

# Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((true_values - forecast_values) / true_values)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape}')

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(true_values, forecast_values))
print(f'Root Mean Squared Error (RMSE): {rmse}')