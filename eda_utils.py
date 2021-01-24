import pandas as pd
import numpy as np
import re
from datetime import datetime
import xml.etree.ElementTree as ET
from itertools import chain
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import svm, model_selection, metrics, cluster
from sklearn.metrics import mean_squared_error
import sklearn
import matplotlib.pyplot as plt
import math
import time
import datetime
import statsmodels.api as sm
import itertools
from matplotlib import pyplot as plt
import warnings
plt.style.use('seaborn-darkgrid')


#########################################################################################################
## --------------------------------------  DATA GENERATION ------------------------------------------- ##
#########################################################################################################

def generateRevenue(resRev, clustDF, dohDF):
    """
    GENERATING RESTAURANT REVENUE DATA USING RESTAURANT REVIEW, CLUSTER, AND NYC DOH DATA
    
    :Inputs
        :param resRev: Restaurant reviews data from the DB
        :param clustDF: Clustered and filtered data according to restaurant in interest
        :param dohDF: NYC Department of Health Data on restaurant grade and inspections

    :Return:
        :return revPivot: Revenue generated dataframe for modeling
    """
    print('\n**** STARTING GENERATION OF REVENUES HISTORICAL DATA ****\n')
    ## PREPROCESS DOH DATA
    print('**** PREPROCESSING DOH DATA ****')
    DOH = dohDF.copy()
    DOH = DOH[['Name', 'ZipCode', 'Date', 'Grade']]
    DOH['Name'] = DOH['Name'].map(lambda x: str(x).lower())

    ## MERGE AND PREPROCESS CLUSTER DATA AND DOH DATA
    print('**** MERGING AND PREPROCESSING DOH AND CLUSTER DATAFRAMES ****')
    resAttrCompl = pd.merge(clustDF, DOH, on = ['Name', 'ZipCode'], how = 'left')
    resAttrCompl = resAttrCompl.drop_duplicates()
    resAttrCompl = resAttrCompl.dropna()

    ## MERGE AND PREPROCESS TO GENERATE REVENUE
    print('**** MERGING ALL DATA TOGETHER AND CALCULATING SALES ****')
    revRevenue = pd.merge(resRev, resAttrCompl, on = ['Name', 'ZipCode', 'Date'], how = 'left')
    revRevenue = revRevenue.drop(columns = 'ZipCode')
    def fineProc(elem):
        try:
            if elem.lower() == 'a':
                return 0.00
            elif elem.lower() == 'b':
                return 553.00
            elif elem.lower() == 'c':
                return 653.37
            else:
                return elem
        except:
            return elem

    def salesProc(elem):
        try:
            if elem.lower() == 'a':
                return 82.86
            elif elem.lower() == 'b':
                return 0.00
            elif elem.lower() == 'c':
                return -143.65
            else:
                return elem
        except:
            return elem

    revRevenue['Fines'] = revRevenue['Grade'].map(lambda x: fineProc(x))
    revRevenue['Fines'] = revRevenue['Fines'].fillna(0)

    revRevenue['Sales'] = revRevenue['Grade'].map(lambda x: salesProc(x))

    revRevenue = revRevenue.drop(columns = ['Grade'])

    revRevenue = revRevenue.fillna(method = 'ffill')
    revRevenue = revRevenue.dropna()
    revRevenue = revRevenue.reset_index(drop = True)

    revRevenue['Review_Count'] = revRevenue['Review_Count'].astype('float64')
    revRevenue['Photos'] = revRevenue['Photos'].astype('float64')
    revRevenue['Fines'] = revRevenue['Fines'].astype('float64')
    revRevenue['Sales'] = revRevenue['Sales'].astype('float64')

    # GENERATE RESTAURANT REVENUE DATA
    revRevenue['Revenue'] = ((revRevenue['Review_Count'] * revRevenue['Review_Vader_Comp'] * revRevenue['Dollars'] * revRevenue['Photos']) - (revRevenue['Fines']) + (revRevenue['Sales']))
    revRevenue = revRevenue.drop_duplicates()

    ## PIVOT REVENUE DATA AND TRANSFORM FOR MODELING
    revPivot = revRevenue.pivot_table(index = ['Date'], columns = 'Name', values = 'Revenue')

    print('**** SALES GENERATION HAS COMPLETED ****')

    return revPivot


#########################################################################################################
## ---------------------------------------  DATA PLOTTING -------------------------------------------- ##
#########################################################################################################

def plotHist(revDF, resName, plotMA = True, plotInterval = True, window = 30, figsize = (15, 5)):
    """
    PLOT HISTORICAL DATA
    :parameters
        :param revDF: Generated restaurant revenue dataframe
        :param resName: Name of restuarnt in interest of analyzing
        :param plotMA: Plot rolling mean, DEFAULT: True
        :param plotInterval: Plot lower and upper bound in between, DEFAULT: True
        :param window: Window for rolling mean and std
        :param figsize: Figure size for plot, DEFAULT: (15, 5)
    """
    print('\n**** PLOTTING HISTORICAL DATA ****')

    revCopy = revDF.copy()
    resName = resName.lower()

    revCopy = revCopy.reset_index()
    revCopy['Date'] = pd.to_datetime(revCopy['Date'], format = '%Y-%m-%d')
    first_idx = revCopy[resName].first_valid_index()
    resRev = revCopy.loc[first_idx:]
    resRev = resRev.groupby('Date')[resName].sum().rename('Revenue')

    rolling_mean = resRev.rolling(window=window).mean()
    rolling_std = resRev.rolling(window=window).std()
    plt.figure(figsize=figsize)
    plt.title('Historical Sales')
    plt.plot(resRev[window:], label='Revenue', color = 'blue')
    if plotMA:
        plt.plot(rolling_mean, 'g', label='MA'+str(window), color="orange")
    if plotInterval:
        lower_bound = rolling_mean - (1.96 * rolling_std)
        upper_bound = rolling_mean + (1.96 * rolling_std)
        plt.fill_between(x=resRev.index, y1=lower_bound, y2=upper_bound, color='grey', alpha=0.4)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def findTrend(revDF, resName, degree = 1, plot = True, figsize = (15, 5)):
    """
    PLOT TREND LINE ON HISTORICAL DATA
    :parameters
        :param revDF: Generated restaurant revenue dataframe
        :param resName: Name of restuarnt in interest of analyzing
        :param plot: Plot the trend line, DEFAULT: True
        :param figsize: Figure size for plot, DEFAULT: (15, 5)
    """
    revCopy = revDF.copy()
    resName = resName.lower()

    revCopy = revCopy.reset_index()
    revCopy['Date'] = pd.to_datetime(revCopy['Date'], format = '%Y-%m-%d')
    first_idx = revCopy[resName].first_valid_index()
    resRev = revCopy.loc[first_idx:]
    resRev = resRev.groupby('Date')[resName].sum().rename('Revenue')

    ## CALCULATE TREND LINE
    print('\n**** CALCULATING TREND LINE ****')
    dtf = resRev.to_frame(name = 'Rev')
    params = np.polyfit(resRev.reset_index().index, resRev.values, deg = degree)
    constant = params[-1]
    dtf['Trend'] = constant
    X = np.array(range(1, len(resRev) + 1))
    for i in range(1, degree+1):
        dtf['Trend'] = dtf['Trend'] + params[i-1]*(X**i)

    ## ACTUAL PLOTTING
    print('**** PLOTTING TREND LINE ****')
    if plot is True:
        ax = dtf.plot(grid = True, title = 'Sales Trend Line', figsize = figsize, color = ['blue', 'orange'])
        ax.set(xlabel = 'Date', ylabel = 'Revenue')
        plt.show()

def findOut(revDF, resName, perc = 0.01, figsize = (15, 5)):
    """
    FIND AND PLOT OUTLIERS FROM REVENUE DATA
    :parameters
        :param revDF: Generated restaurant revenue dataframe
        :param resName: Name of restuarnt in interest of analyzing
        :param perc: 
        :param figsize: Figure size for plot, DEFAULT: (15, 5)
    """
    print('\n**** FINDING AND PLOTTING OUTLIERS IN DATAFRAME ****')
    revCopy = revDF.copy()
    resName = resName.lower()

    revCopy = revCopy.reset_index()
    revCopy['Date'] = pd.to_datetime(revCopy['Date'], format = '%Y-%m-%d')
    first_idx = revCopy[resName].first_valid_index()
    resRev = revCopy.loc[first_idx:]
    resRev = resRev.groupby('Date')[resName].sum().rename('Revenue')

    ## FIT SVM
    scaler = StandardScaler()
    revDFScaled = scaler.fit_transform(resRev.values.reshape(-1, 1))
    model = svm.OneClassSVM(nu = perc, kernel = 'rbf', gamma = 0.01)
    model.fit(revDFScaled)

    ## DTF OUTPUT OF OUTLIERS
    dtf_outliers = resRev.to_frame('Rev')
    dtf_outliers['Index'] = range(len(resRev))
    dtf_outliers['Outlier'] = model.predict(revDFScaled)
    dtf_outliers['Outlier'] = dtf_outliers['Outlier'].apply(lambda x: 1 if x == -1 else 0)

    ## ACTUAL PLOTTING
    fig, ax = plt.subplots(figsize = figsize)
    ax.set(title = f"Sales Outlier Detection: Found {sum(dtf_outliers['Outlier'] == 1)}")
    ax.plot(dtf_outliers['Index'], dtf_outliers['Rev'], color = 'blue')
    ax.scatter(x = dtf_outliers[dtf_outliers['Outlier']==1]['Index'], y = dtf_outliers[dtf_outliers['Outlier']==1]['Rev'], color = 'red')
    ax.grid(True)
    plt.show()