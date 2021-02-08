# SeaPlane_travel

import pandas as pd
import numpy as np
from datetime import date
from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams

df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Edureka_Notebook\\M13\\SeaPlaneTravel.csv')
df.head()

df['Month'] = pd.to_datetime(df['Month'])
df1=df.rename(columns={'Month':'Date','#Passengers':'Passengers'})
df1.head()

indexed_df = df1.set_index('Date')
ts_month = indexed_df.Passengers
ts_month.head(5)

plt.plot(ts_month)

def test_stationarity(timeseries):
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')   #display color lines with labels
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # perform Dickey-Fuller Test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries,autolag='AIC')
    dftest_output = pd.Series(dftest[0:4],index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dftest_output['Critical Value (%s)'%key] = value
    print(dftest_output)
    
test_stationarity(ts_month)

ts_month_log = np.log(ts_month)
ts_month_log_diff = ts_month_log-ts_month_log.shift()
plt.plot(ts_month_log_diff)

ts_month_log_diff.dropna(inplace=True)
test_stationarity(ts_month_log_diff)

lag_acf = acf(ts_month_log_diff,nlags=10)
lag_pacf = pacf(ts_month_log_diff,nlags=10,method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-7.96/np.sqrt(len(ts_month_log_diff)),linestyle='--',color='gray')
plt.axhline(y=7.96/np.sqrt(len(ts_month_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-7.96/np.sqrt(len(ts_month_log_diff)),linestyle='--',color='gray')
plt.axhline(y=7.96/np.sqrt(len(ts_month_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

model = ARIMA(ts_month_log,order=(2,1,2))
result_arima = model.fit(disp=1)
plt.plot(ts_month_log_diff)
plt.plot(result_arima.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((result_arima.fittedvalues-ts_month_log_diff)**2))

print(result_arima.summary())
# plot residual errors
residuals = pd.DataFrame(result_arima.resid)
residuals.plot(kind='kde')
print(residuals.describe())

predictions_ARIMA_diff = pd.Series(result_arima.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_month_log.ix[0], index=ts_month_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts_month)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts_month)**2)/len(ts_month)))

size = int(len(ts_month_log)-15)
train,test = ts_month_log[0:size],ts_month_log[size:len(ts_month_log)]
history = [x for x in train]
predictions = list()
print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(history,order=(2,1,2))
    model_fit=model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
print('predicted=%f, expected=%f' % (np.exp(yhat), np.exp(obs)))

error = mean_squared_error(test, predictions)
print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test MSE: %.6f' % error)
predictions_series = pd.Series(predictions, index = test.index)

fig, ax = plt.subplots()
ax.set(title='Spot Exchange Rate, Euro into USD', xlabel='Date', ylabel='Euro into USD')
ax.plot(ts_month[-60:], 'o', label='observed')
ax.plot(np.exp(predictions_series), 'g', label='rolling one-step out-of-sample forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
