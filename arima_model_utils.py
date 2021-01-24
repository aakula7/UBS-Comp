import pandas as pd
import numpy as np
import re
from datetime import datetime
import xml.etree.ElementTree as ET
from itertools import chain
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sklearn
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import itertools
import pmdarima
import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-darkgrid')


#########################################################################################################
## ---------------------------------------  ARIMA MODELING ------------------------------------------- ##
#########################################################################################################

def arimaModeling(revDF, resName, resample = 'MS', model = 'additive', s = 12, max_p = 3, max_d = 3, max_q = 3, max_P = 3, max_D = 3, max_Q = 3, seasonal = True, stationary = False, figsize = (15, 5)):
    """
    FORECAST FUTURE SALES WITH THE USE OF ARIMA MODELING

    Inputs:
        :param revDF: Generated and clustered restaurant revenue dataframe
        :param resName: Name of restuarnt in interest of analyzing
        :param resample: Frequency conversion and resampling of time series
        :param model: Type of seasonal decompose model
        :param s: Number of time steps for a single season period, DEFAULT: 12
        :param max_p: Lag order, DEFAULT: 3
        :param max_d: Degree of differencing, DEFAULT: 3
        :param max_q: Order of the moving average, DEFAULT: 3
        :param max_P: Seasonal autoregressive order, DEFAULT: 3
        :param max_D: Seasonal difference order, DEFAULT: 3
        :param max_Q: Seasonal moving average order, DEFAULT: 3
        :param seasonal: Whether to fit a seasonal ARIMA
        :param stationary: Whether the time-series is stationary
        :param figsize: Plot figure size
    """

    revCopy = revDF.copy()
    resName = resName.lower()

    revCopy = revCopy.reset_index()
    revCopy['Date'] = pd.to_datetime(revCopy['Date'], format = '%Y-%m-%d')
    first_idx = revCopy[resName].first_valid_index()
    resRev = revCopy.loc[first_idx:]
    resRev = resRev.reset_index(drop = True)
    resRev = resRev.groupby('Date').sum()

    reSamp = resRev[resName].resample(resample).mean()
    reSamp = reSamp.fillna(0)

    plt.figure(figsize = figsize)
    decomposition = sm.tsa.seasonal_decompose(reSamp, model = model)
    decomposition.plot()
    plt.show()

    print('\n **** EVALUATING BEST ARIMA PARAMETERS FOR PREDICTION AND FORECASTING ****')
    best_model = pmdarima.auto_arima(reSamp, seasonal = seasonal, stationary = stationary, m = s, 
                                        information_criterion = 'aic', max_order = 20, max_p = max_p,
                                        max_d = max_d, max_q = max_q, max_P = max_P, max_D = max_D, 
                                        max_Q = max_Q, error_action = 'ignore')
    print(f'Best Model --> (p, d, q): {best_model.order} and (P, D, Q, s): {best_model.seasonal_order}')
    print(best_model.summary())

    print('\n **** BUILDING AND FITTING ARIMA MODEL WITH SELECTED PARAMETERS ****')
    mod = sm.tsa.statespace.SARIMAX(reSamp,
                                order=(best_model.order[0], best_model.order[1], best_model.order[2]),
                                seasonal_order=(best_model.seasonal_order[0], 
                                                best_model.seasonal_order[1], 
                                                best_model.seasonal_order[2],
                                                best_model.seasonal_order[3]),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()

    ## PREDICT ON BOTTOM 30% OF OBSERVED VALUES
    print('\n **** PREDICTING ON BOTTOM 30% OF OBSERVED VALUES ****')
    pred = results.get_prediction(start=reSamp.index[int(len(reSamp)*-0.3)], dynamic=False)
    plt.figure(figsize = figsize)
    ax0 = plt.subplot2grid((1,3), (0,0), rowspan=1, colspan=2)
    pred_ci = pred.conf_int()
    ax0 = reSamp[str(reSamp.index[0].year):].plot(label='Observed', color = 'blue', grid = True, title = 'History & Prediction')
    pred.predicted_mean.plot(ax=ax0, color = 'red', label='Prediction', alpha=.7)
    ax0.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax0.set(xlabel='Date')
    ax0.set(ylabel = 'Sales')
    plt.legend()
    plt.grid(True)

    ## ZOOM IN ON PREDICTION
    ax1 = plt.subplot2grid((1,3), (0,2), rowspan=1, colspan=1)
    first_idx = reSamp.index[int(len(reSamp)*-0.3)]
    first_loc = reSamp.index.tolist().index(first_idx)
    zoom_idx = reSamp.index[first_loc]
    ax1 = reSamp.loc[zoom_idx:].plot(color='blue', label='Observed', grid=True, title="Zoom on the Prediction")
    pred.predicted_mean.loc[zoom_idx:].plot(ax=ax1, color = 'red', label='Prediction', alpha=.7)
    ax1.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax1.set(xlabel='Date')
    plt.legend()
    plt.grid(True)
    plt.show()

    ## EVALUATE PREDICTIONS
    y_forecasted = pred.predicted_mean
    y_truth = reSamp[int(len(reSamp)*-0.3):]
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print(f'The Mean Squared Error of our forecasts is {round(mse, 2)}')
    print(f'The Root Mean Squared Error of our forecasts is {round(np.sqrt(mse), 2)} \n')

    ## FORECAST 2 YEARS OF SALES
    print('\n **** FORECASTING NEXT 2 YEARS OF SALES ****')
    pred_uc = results.get_forecast(steps=24)
    pred_ci = pred_uc.conf_int()
    plt.figure(figsize = figsize)
    ax0 = plt.subplot2grid((1,3), (0,0), rowspan=1, colspan=2)
    ax0 = reSamp.plot(label='Observed', color='blue', title = 'History & Forecast')
    pred_uc.predicted_mean.plot(ax=ax0, label='Forecast', color = 'red')
    ax0.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax0.set_xlabel('Date')
    ax0.set_ylabel('Sales')
    plt.legend()
    plt.grid(True)

    ## ZOOM INTO FORECAST
    ax1 = plt.subplot2grid((1,3), (0,2), rowspan=1, colspan=1)
    first_idx = reSamp.index[int(len(reSamp)*-0.1)]
    first_loc = reSamp.index.tolist().index(first_idx)
    zoom_idx = reSamp.index[first_loc]
    ax1 = reSamp.loc[zoom_idx:].plot(color='blue', label='Observed', grid=True, title="Zoom on the Forecast")
    pred_uc.predicted_mean.loc[zoom_idx:].plot(ax=ax1, color = 'red', label='Forecast', alpha=.7)
    ax1.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax1.set(xlabel='Date')
    plt.legend()
    plt.grid(True)
    plt.show()

    print('\n **** SAVING ARIMA MODEL PREDICTIONS LOCALLY ****')
    resFileName = resName.replace(' ', '_')
    fileName = f'{resFileName.upper()}_ARIMA_PREDICTIONS.csv'
    pred.predicted_mean.to_csv(fileName)

    print('\n **** SAVING ARIMA MODEL FORECASTING LOCALLY ****')
    resFileName = resName.replace(' ', '_')
    fileName = f'{resFileName.upper()}_ARIMA_FORECASTING.csv'
    pred_uc.predicted_mean.to_csv(fileName)