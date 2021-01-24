import pandas as pd
import numpy as np
import re
from datetime import datetime
import xml.etree.ElementTree as ET
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-darkgrid')


#########################################################################################################
## --------------------------------------  DATA PREPERATION ------------------------------------------ ##
#########################################################################################################

def split_train_test(revPivDF, valid = None, test = 0.25, plot = True, figsize = (15,5)):
    """
    SPLIT TRAIN AND TEST HISTORICAL DATA FROM ANY GIVEN DATA POINT
    :parameters
        :param revPivotDF: Restaurant revenue dataframe generated
        :param valid: Valid data split level
        :param test: Test data split level, DEFAULT: 0.25
        :param plot: If the function should plot the split, DEFAULT: True
        :pram figsize: Figure size of plot
    """
    if type(test) is float:
        split = int(len(revPivDF)*(1-test))
        perc = test
    elif type(test) is str:
        split = revPivDF.reset_index()[revPivDF.reset_index().iloc[:,0]==test].index[0]
        perc = round(len(revPivDF[split:])/len(revPivDF), 2)
    else:
        split = test
        perc = round(len(revPivDF[split:])/len(revPivDF), 2)

    if valid is None:
        revPivDF_train = revPivDF[:split]
        revPivDF_test = revPivDF[split:]
        print(f'--- Splitting at index: {split} ---|--- Splitting train and test value: {revPivDF.index[split]} ---|--- Test size percentage: {perc} ---')
        if plot is True:
          plt.figure(figsize = figsize)
          fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=figsize)
          revPivDF_train.plot(ax=ax[0], grid=True, title="Train", color="blue")
          revPivDF_test.plot(ax=ax[1], grid=True, title="Test", color="blue")
          ax[0].set(xlabel='Date')
          ax[0].set(ylabel = 'Daily Sales')
          ax[1].set(xlabel='Date')
          plt.show()
        else:
          pass

        return (revPivDF_train, revPivDF_test)
    else:
        test_len = int(len(revPivDF)*perc)
        valid_len = int(len(revPivDF)*valid)
        train_len = int(len(revPivDF)-(test_len+valid_len))
        revPivDF_train = revPivDF[:train_len]
        revPivDF_valid = revPivDF[train_len:train_len+valid_len]
        revPivDF_test = revPivDF[train_len+valid_len:]
        print(f'--- Splitting at index: {split} ---|--- Splitting train and valid value: {revPivDF.index[train_len]} ---|--- Valid size percentage: {valid} ---|--- Splitting valid and test value: {revPivDF.index[train_len+valid_len]} ---|--- Test size percentage: {perc} ---')
        if plot is True:
          plt.figure(figsize = figsize)
          fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=figsize)
          revPivDF_train.plot(ax=ax[0], grid=True, title="Train", color = 'blue')
          revPivDF_valid.plot(ax=ax[1], grid=True, title="Valid", color = 'blue')
          revPivDF_test.plot(ax=ax[2], grid=True, title="Test", color = 'blue')
          ax[0].set(xlabel='Date')
          ax[0].set(ylabel = 'Daily Sales')
          ax[1].set(xlabel='Date')
          ax[2].set(xlabel='Date')
          plt.show()
        else:
          pass

        return (revPivDF_train, revPivDF_valid, revPivDF_test)


#########################################################################################################
## -----------------------------------  LSTM ANOMALLY DETECTION -------------------------------------- ##
#########################################################################################################
def anomallyDet(revDF, resName, valid = None, test = 0.25, plot = True, figsize = (15, 5), n_steps = 30, units = 64, dropout = 0.2, optimizer = 'adam', metrics = 'accuracy', batch_size = 32, loss = 'mae', epochs = 100):
    """
    FORECAST FUTURE SALES WITH THE USE OF ARIMA MODELING

    Inputs:
        :param revDF: Generated and clustered restaurant revenue dataframe
        :param resName: Name of restuarnt in interest of analyzing
        :param valid: Valid dataframe size
        :param test: Test dataframe size, DEFAULT: 0.25
        :param plot: If the function should plot, DEFAULT: True
        :param figsize: Plot figure size, DEFAULT: (15, 5)
        :param n_steps: Sequence for n_steps of days for historical data
        :param units: Dimensionality of the output space
        :param dropout: Fraction of the units to drop for the linear transformation of the inputs, DEFAULT = 0.2
        :param optimizier: Updating modle in response to the output of the loss function, DEFAULT: adam
        :param loss: Compute the quantity that a model should seek to minimize, DEFAULT: mae
        :param metrics: Function used to judge the performance of the LSTM model ,DEFAULT: accuracy
        :param batch_size: Number of samples per gradient update, DEFAULT: 32
        :param epochs: Number of epochs to train LSTM model, DEFAULT: 100
    """

    revCopy = revDF.copy()
    resName = resName.lower()

    print('**** SPLICING GENERATED DATAFRAME ****')
    revCopy = revCopy.reset_index()
    revCopy = revCopy[['Date', resName]]
    revCopy['Date'] = revCopy['Date'].astype('datetime64')
    first_idx = revCopy[resName].first_valid_index()
    revCopy = revCopy.loc[first_idx:]
    revCopy = revCopy.reset_index(drop = True)
    revCopy = revCopy.groupby('Date').sum()

    if valid is None:
        print('**** SPLITING INTO TRAIN AND TEST **** \n')
        trainDF, testDF = split_train_test(revCopy, valid = valid, test = test, plot = plot, figsize = figsize)
        
        print('**** ROBUST SCALING TRAIN AND TEST DATA **** \n')
        robust = RobustScaler(quantile_range=(25, 75)).fit(trainDF)
        trainDF_scaled = robust.transform(trainDF)
        testDF_scaled = robust.transform(testDF)
        
        ## HELPER FUNCTION
        def create_dataset(X, y, time_steps=1):
            a, b = [], []
            for i in range(len(X) - time_steps):
                v = X[i:(i + time_steps)]
                a.append(v)
                b.append(y[i + time_steps])
            return np.array(a), np.array(b)

        ## CREATE SEQUENCES WITH N_STEPS DAYS OF HISTORICAL DATA
        n_steps = n_steps

        print('**** RESHAPING DATA INTO 3D FOR LSTM MODEL **** \n')
        ## RESHAPE TO 3D [n_samples, n_steps, n_features]
        X_train, y_train = create_dataset(trainDF_scaled, trainDF_scaled, n_steps)
        X_test, y_test = create_dataset(testDF_scaled, testDF_scaled, n_steps)
        print('X_train shape:', X_train.shape)
        print('y_train:', y_train.shape)
        print('X_test shape:', X_test.shape)
        print('y_test:', y_test.shape)

        print('**** BUILDING LSTM MODEL ****')
        units = units; dropout = dropout; optimizer = optimizer; loss = loss; epochs = epochs
        model = Sequential()
        model.add(LSTM(units = units, input_shape = (X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(rate = dropout))
        model.add(RepeatVector(n = X_train.shape[1]))
        model.add(LSTM(units = units, return_sequences = True))
        model.add(Dropout(rate = dropout))
        model.add(TimeDistributed(Dense(units = X_train.shape[2])))
        print(model.summary())

        print('\n **** COMPILING AND FITTING LSTM MODEL **** \n')
        model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split = 0.2, shuffle = False)
        
        if plot is True:
            print('**** PLOT MODEL LOSS OVER EPOCHS ****')
            plt.figure(figsize = figsize)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='best')
            plt.grid(True)
            plt.show()
        else:
            pass

        print('\n **** PREDICTING ON TEST DATAFRAME ****')
        y_pred = model.predict(X_test)
        mae = np.mean(np.abs(y_pred - X_test), axis=1)
        ## RESHAPING PREDICTION
        pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1]), y_pred.shape[2])
        ## RESHAPING TEST DATA
        X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1]), X_test.shape[2])
        ## ERROR COMPUTATION
        errors = X_test - pred
        print('Error:', errors.shape)
        ## RMSE DATA
        RMSE = math.sqrt(mean_squared_error(X_test, pred))
        print(f'Test RMSE: {RMSE} \n')

        ## DETECTING ANOMALIES
        print('**** DETECTING ANOMALIES IN SALES ****')
        dist = np.linalg.norm(X_test - pred, axis = 1)
        scores = dist.copy()
        scores.sort()
        cut_off = int(0.8 * len(scores))
        threshold = scores[cut_off]
        score = pd.DataFrame(index = testDF[n_steps:].index)
        score['Loss'] = mae
        score['Threshold'] = threshold
        score['Anomaly'] = score['Loss'] > score['Threshold']
        score[resName] = testDF[n_steps:][resName]
        anomalies = score[score['Anomaly'] == True]
        x = pd.DataFrame(anomalies[resName])
        x = pd.DataFrame(robust.inverse_transform(x))
        x.index = anomalies.index
        x.rename(columns = {0: 'Revenue'}, inplace = True)
        anomalies = anomalies.join(x, how = 'left')
        anomalies = anomalies.drop(columns = [resName], axis = 1)

        test_inv = pd.DataFrame(robust.inverse_transform(testDF[n_steps:]))
        test_inv.index = testDF[n_steps:].index
        test_inv.rename(columns = {0 : resName}, inplace = True)

        if plot is True:
            print('**** PLOTTING ANOMALLY DETECTION ****')
            plt.figure(figsize = figsize)
            plt.plot(test_inv.index, test_inv[resName], color = 'gray', label = resName)
            sns.scatterplot(anomalies.index, anomalies['Revenue'], color = 'red', s = 55, label = 'Anomaly')
            plt.xticks(rotation = 90)
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.legend(loc = 'best')
            plt.grid(True)
            plt.show()

            print('\n **** SAVING ANOMALY MODEL PREDICTIONS LOCALLY ****')
            resFileName = resName.replace(' ', '_')
            fileName = f'{resFileName.upper()}_ANOMALY_PREDICTIONS.csv'
            anomalies.to_csv(fileName)
        else:
            print('\n **** SAVING ANOMALY MODEL PREDICTIONS LOCALLY ****')
            resFileName = resName.replace(' ', '_')
            fileName = f'{resFileName.upper()}_ANOMALY_PREDICTIONS.csv'
            anomalies.to_csv(fileName)
        
    else:
        print('**** SPLITING INTO TRAIN, VALID, AND TEST **** \n')
        trainDF, validDF, testDF = split_train_test(revCopy, valid = valid, test = test, plot = plot, figsize = figsize)
        
        print('**** ROBUST SCALING TRAIN, VALID, TEST DATA **** \n')
        robust = RobustScaler(quantile_range=(25, 75)).fit(trainDF)
        trainDF_scaled = robust.transform(trainDF)
        validDF_scaled = robust.transform(validDF)
        testDF_scaled = robust.transform(testDF)
        
        ## HELPER FUNCTION
        def create_dataset(X, y, time_steps=1):
            a, b = [], []
            for i in range(len(X) - time_steps):
                v = X[i:(i + time_steps)]
                a.append(v)
                b.append(y[i + time_steps])
            return np.array(a), np.array(b)

        ## CREATE SEQUENCES WITH N_STEPS DAYS OF HISTORICAL DATA
        n_steps = n_steps

        print('**** RESHAPING DATA INTO 3D FOR LSTM MODEL **** \n')
        ## RESHAPE TO 3D [n_samples, n_steps, n_features]
        X_train, y_train = create_dataset(trainDF_scaled, trainDF_scaled, n_steps)
        X_valid, y_valid = create_dataset(validDF_scaled, validDF_scaled, n_steps)
        X_test, y_test = create_dataset(testDF_scaled, testDF_scaled, n_steps)
        print('X_train shape:', X_train.shape)
        print('y_train:', y_train.shape)
        print('X_test shape:', X_test.shape)
        print('y_test:', y_test.shape)

        print('**** BUILDING LSTM MODEL ****')
        units = units; dropout = dropout; optimizer = optimizer; loss = loss; epochs = epochs
        model = Sequential()
        model.add(LSTM(units = units, input_shape = (X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(rate = dropout))
        model.add(RepeatVector(n = X_train.shape[1]))
        model.add(LSTM(units = units, return_sequences = True))
        model.add(Dropout(rate = dropout))
        model.add(TimeDistributed(Dense(units = X_train.shape[2])))
        print(model.summary())

        print('\n **** COMPILING AND FITTING LSTM MODEL **** \n')
        model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_valid, y_valid), shuffle = False)
        
        if plot is True:
            print('**** PLOT MODEL LOSS OVER EPOCHS ****')
            plt.figure(figsize = figsize)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='best')
            plt.grid(True)
            plt.show()
        else:
            pass

        print('\n **** PREDICTING ON TEST DATAFRAME ****')
        y_pred = model.predict(X_test)
        mae = np.mean(np.abs(y_pred - X_test), axis=1)
        ## RESHAPING PREDICTION
        pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1]), y_pred.shape[2])
        ## RESHAPING TEST DATA
        X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1]), X_test.shape[2])
        ## ERROR COMPUTATION
        errors = X_test - pred
        print('Error:', errors.shape)
        ## RMSE DATA
        RMSE = math.sqrt(mean_squared_error(X_test, pred))
        print(F'Test RMSE: {RMSE}')

        ## DETECTING ANOMALIES
        print('\n **** DETECTING ANOMALIES IN SALES ****')
        dist = np.linalg.norm(X_test - pred, axis = 1)
        scores = dist.copy()
        scores.sort()
        cut_off = int(0.8 * len(scores))
        threshold = scores[cut_off]
        score = pd.DataFrame(index = testDF[n_steps:].index)
        score['Loss'] = mae
        score['Threshold'] = threshold
        score['Anomaly'] = score['Loss'] > score['Threshold']
        score[resName] = testDF[n_steps:][resName]
        anomalies = score[score['Anomaly'] == True]
        x = pd.DataFrame(anomalies[resName])
        x = pd.DataFrame(robust.inverse_transform(x))
        x.index = anomalies.index
        x.rename(columns = {0: 'Revenue'}, inplace = True)
        anomalies = anomalies.join(x, how = 'left')
        anomalies = anomalies.drop(columns = [resName], axis = 1)

        test_inv = pd.DataFrame(robust.inverse_transform(testDF[n_steps:]))
        test_inv.index = testDF[n_steps:].index
        test_inv.rename(columns = {0 : resName}, inplace = True)

        if plot is True:
            print('**** PLOTTING ANOMALLY DETECTION ****')
            plt.figure(figsize = figsize)
            plt.plot(test_inv.index, test_inv[resName], color = 'gray', label = resName)
            sns.scatterplot(anomalies.index, anomalies['Revenue'], color = 'red', s = 55, label = 'Anomaly')
            plt.xticks(rotation = 90)
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.legend(loc = 'best')
            plt.grid(True)
            plt.show()

            print('\n **** SAVING ANOMALY MODEL PREDICTIONS LOCALLY ****')
            resFileName = resName.replace(' ', '_')
            fileName = f'{resFileName.upper()}_ANOMALY_PREDICTIONS.csv'
            anomalies.to_csv(fileName)
        else:
            print('\n **** SAVING ANOMALY MODEL PREDICTIONS LOCALLY ****')
            resFileName = resName.replace(' ', '_')
            fileName = f'{resFileName.upper()}_ANOMALY_PREDICTIONS.csv'
            anomalies.to_csv(fileName)


#########################################################################################################
## --------------------------------  LSTM MULTIVARIATE PREDICTION ------------------------------------ ##
#########################################################################################################
def multiVariate(revDF, resName, valid = None, test = 0.25, plot = True, figsize = (15, 5), look_back = 3, dropout = 0.2, lstmAct = 'softmax', denseAct = 'sigmoid', optimizer = 'adam', metrics = 'accuracy', batch_size = 32, loss = 'mae', epochs = 100):
    """
    FORECAST FUTURE SALES WITH THE USE OF ARIMA MODELING

    Inputs:
        :param revDF: Generated and clustered restaurant revenue dataframe
        :param resName: Name of restuarnt in interest of analyzing
        :param valid: Valid dataframe size
        :param test: Test dataframe size, DEFAULT: 0.25
        :param plot: If the function should plot, DEFAULT: False
        :param figsize: Plot figure size, DEFAULT: (15, 5)
        :param look_back: How far behind to look for evaluating model, DEFAULT: 3
        :param dropout: Fraction of the units to drop for the linear transformation of the inputs, DEFAULT = 0.2
        :param lstmAct: Applies activation function on LSTM layer of model
        :param denseAct: Applies activation function on Dense layer of model
        :param optimizier: Updating modle in response to the output of the loss function, DEFAULT: adam
        :param loss: Compute the quantity that a model should seek to minimize, DEFAULT: mae
        :param metrics: Function used to judge the performance of the LSTM model ,DEFAULT: accuracy
        :param batch_size: Number of samples per gradient update, DEFAULT: 32
        :param epochs: Number of epochs to train LSTM model, DEFAULT: 100
    """

    revCopy = revDF.copy()
    resName = resName.lower()

    print('**** SPLICING GENERATED DATAFRAME **** \n')
    revCopy = revCopy.reset_index()
    first_idx = revCopy[resName].first_valid_index()
    revCopy = revCopy.loc[first_idx:]
    revCopy = revCopy.reset_index(drop = True)
    revCopy = revCopy.fillna(0)
    revCopy['Date'] = revCopy['Date'].astype('datetime64')
    revCopy = revCopy.set_index('Date')

    ## REARRANGE DATAFRAME TO PREDENT RESTAURANT
    print('**** REARRANGING DATAFRAME TO PREDICT ON RESTAURANT OF INTEREST **** \n')
    colsArrg = [resName]
    for col in revCopy.columns:
        if col not in colsArrg:
            colsArrg.append(col)
    revCopy = revCopy[colsArrg]
    
    resCopy_values = revCopy.values

    print('**** MIN MAX SCALING DATA VALUES **** \n')   
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled = scaler.fit_transform(resCopy_values)
    scaled = pd.DataFrame(scaled)
    
    ## HELPER FUNCTION
    def ts(scaledDF, look_back = look_back, pred_col = 0):
        """
        CREATING MULTIVARIATE TIMESERIES DATA

        Inputs:
          :param scaledDF: Dataframe that has been scaled
          :param look_back: Value to look back for crating timeseries data
          :param pred_col: Column to predict on

        Return:
          :return final_df: Dataframe that will be passed for modeling
        """
        t = scaledDF.copy()
        t['id'] = range(1, len(t) + 1)
        t = t.iloc[:-look_back, :]
        t.set_index('id', inplace = True)
        
        pred_value = scaledDF.copy()
        pred_value = pred_value.iloc[look_back:, pred_col]
        pred_value.columns = ['Pred']
        pred_value = pd.DataFrame(pred_value)

        pred_value['id'] = range(1, len(pred_value) + 1)
        pred_value.set_index('id', inplace = True)
        final_df = pd.concat([t, pred_value], axis = 1)

        return final_df

    arrDF = ts(scaled, look_back = look_back, pred_col = 0)
    arrDF.fillna(0, inplace = True)

    colNames = [col for col in arrDF.columns[:-1]]
    colNames.append('v0(t)')
    arrDF.columns = colNames

    arrDF_values = arrDF.values
    
    if valid is None:
        print('**** SPLITING INTO TRAIN AND TEST **** \n')
        test_sample = int(len(arrDF) * test)
        trainDF = arrDF_values[:-test_sample, :]
        testDF = arrDF_values[test_sample:, :]

        X_train, y_train = trainDF[:,:-1], trainDF[:,-1]
        X_test, y_test = testDF[:,:-1], testDF[:,-1]

        print('**** RESHAPING DATA INTO 3D FOR LSTM MODEL **** \n')
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        print('**** BUILDING LSTM MODEL ****')
        model = Sequential()
        model.add(LSTM(50, activation = lstmAct, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 30,activation = lstmAct, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 30, activation = lstmAct))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1, activation = denseAct))
        print(model.summary())

        print('\n **** COMPILING AND FITTING LSTM MODEL ****')
        model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_test, y_test), shuffle = False)

        if plot is True:
            print('\n **** PLOT MODEL LOSS OVER EPOCHS ****')
            plt.figure(figsize = figsize)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='best')
            plt.grid(True)
            plt.show()
        else:
            pass

        if plot is True:
            print('\n **** PLOT MODEL ACCURACY OVER EPOCHS ****')
            plt.figure(figsize = figsize)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='best')
            plt.grid(True)
            plt.show()
        else:
            pass

        print('\n **** PREDICTING ON TEST DATAFRAME ****')
        yhat = model.predict(X_test)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))

        inv_yhat = np.concatenate((yhat, X_test[:, 1:]), axis = 1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]

        y_test = y_test.reshape((len(y_test), 1))
        inv_y = np.concatenate((y_test, X_test[:,1:]), axis = 1)
        inv_y = inv_y[:,0]

        preds = pd.DataFrame(data = inv_y, index = revCopy.index[-len(inv_y):], columns = [f'{resName} actual'])
        preds[f'{resName} prediction'] = inv_yhat

        print('\n **** EVALUATING ON TEST DATAFRAME ****')
        print(f'Mean Absolute Error(MAE): {sklearn.metrics.mean_absolute_error(inv_y, inv_yhat)}')
        print(f'Mean Squared Error(MSE): {sklearn.metrics.mean_squared_error(inv_y, inv_yhat)}')
        print(f'Root Mean Squared Error (RMSE): {math.sqrt(sklearn.metrics.mean_squared_error(inv_y, inv_yhat))}')
        print(f'R Square (R^2): {sklearn.metrics.r2_score(inv_y, inv_yhat)} \n')

        if plot is True:
            print('**** PLOT MODEL PREDICTION ****')
            revRes = revCopy[resName]
            plt.figure(figsize = (15, 5))
            ax0 = plt.subplot2grid((1,3), (0,0), rowspan=1, colspan=2)
            ax0 = revRes[str(revRes.index[0].year):].plot(label='Observed', color = 'blue', grid = True, title = 'History & Prediction')
            preds[f'{resName} prediction'].plot(ax=ax0, color = 'red', label='Prediction', alpha=.7)
            ax0.set(xlabel='Date')
            ax0.set(ylabel = 'Sales')
            plt.legend()
            plt.grid(True)

            ## ZOOM IN ON PREDICTION
            ax1 = plt.subplot2grid((1,3), (0,2), rowspan=1, colspan=1)
            ax1 = preds[f'{resName} actual'].plot(color='blue', label='Observed', grid=True, title="Zoom on the Prediction")
            preds[f'{resName} prediction'].plot(ax=ax1, color = 'red', label='Prediction', alpha=.7)
            ax1.set(xlabel='Date')
            plt.legend()
            plt.grid(True)
            plt.show()
            print('\n **** SAVING MULTIVARIATE MODEL PREDICTIONS LOCALLY ****')
            resFileName = resName.replace(' ', '_')
            fileName = f'{resFileName.upper()}_PREDICTIONS.csv'
            preds.to_csv(fileName)
        else:
            print('\n **** SAVING MULTIVARIATE MODEL PREDICTIONS LOCALLY ****')
            resFileName = resName.replace(' ', '_')
            fileName = f'{resFileName.upper()}_PREDICTIONS.csv'
            preds.to_csv(fileName)
    
    else:
        print('**** SPLITING INTO TRAIN, VALID, AND TEST **** \n')
        test_sample = int(len(arrDF) * test)
        valid_sample = int(len(arrDF) * valid)
        train_sample = int(len(arrDF)-(valid_sample + test_sample))
        
        trainDF = arrDF_values[:train_sample, :]
        validDF = arrDF_values[train_sample:train_sample+valid_sample, :]
        testDF = arrDF_values[train_sample+valid_sample:, :]

        X_train, y_train = trainDF[:,:-1], trainDF[:,-1]
        X_valid, y_valid = validDF[:,:-1], validDF[:,:-1]
        X_test, y_test = testDF[:,:-1], testDF[:,-1]

        print('**** RESHAPING DATA INTO 3D FOR LSTM MODEL **** \n')
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        print('**** BUILDING LSTM MODEL ****')
        model = Sequential()
        model.add(LSTM(50, activation = lstmAct, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 30,activation = lstmAct, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 30, activation = lstmAct))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1, activation = denseAct))
        print(model.summary())

        print('\n **** COMPILING AND FITTING LSTM MODEL ****')
        model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_valid, X_valid), shuffle = False)

        if plot is True:
            print('\n **** PLOT MODEL LOSS OVER EPOCHS ****')
            plt.figure(figsize = figsize)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='best')
            plt.grid(True)
            plt.show()
        else:
            pass

        if plot is True:
            print('\n **** PLOT MODEL ACCURACY OVER EPOCHS ****')
            plt.figure(figsize = figsize)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='best')
            plt.grid(True)
            plt.show()
        else:
            pass

        print('\n **** PREDICTING ON TEST DATAFRAME **** \n')
        yhat = model.predict(X_test)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))

        inv_yhat = np.concatenate((yhat, X_test[:, 1:]), axis = 1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]

        y_test = y_test.reshape((len(y_test), 1))
        inv_y = np.concatenate((y_test, X_test[:,1:]), axis = 1)
        inv_y = inv_y[:,0]

        preds = pd.DataFrame(data = inv_y, index = revCopy.index[-len(inv_y):], columns = [f'{resName} actual'])
        preds[f'{resName} prediction'] = inv_yhat

        print('\n **** EVALUATING ON TEST DATAFRAME ****')
        print(f'Mean Absolute Error(MAE): {sklearn.metrics.mean_absolute_error(inv_y, inv_yhat)}')
        print(f'Mean Squared Error(MSE): {sklearn.metrics.mean_squared_error(inv_y, inv_yhat)}')
        print(f'Root Mean Squared Error (RMSE): {math.sqrt(sklearn.metrics.mean_squared_error(inv_y, inv_yhat))}')
        print(f'R Square (R^2): {sklearn.metrics.r2_score(inv_y, inv_yhat)} \n')

        if plot is True:
            print('**** PLOT MODEL PREDICTING ****')
            revRes = revCopy[resName]
            plt.figure(figsize = (15, 5))
            ax0 = plt.subplot2grid((1,3), (0,0), rowspan=1, colspan=2)
            ax0 = revRes[str(revRes.index[0].year):].plot(label='Observed', color = 'blue', grid = True, title = 'History & Prediction')
            preds[f'{resName} prediction'].plot(ax=ax0, color = 'red', label='Prediction', alpha=.7)
            ax0.set(xlabel='Date')
            ax0.set(ylabel = 'Sales')
            plt.legend()
            plt.grid(True)

            ## ZOOM IN ON PREDICTION
            ax1 = plt.subplot2grid((1,3), (0,2), rowspan=1, colspan=1)
            ax1 = preds[f'{resName} actual'].plot(color='blue', label='Observed', grid=True, title="Zoom on the Prediction")
            preds[f'{resName} prediction'].plot(ax=ax1, color = 'red', label='Prediction', alpha=.7)
            ax1.set(xlabel='Date')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            print('\n **** SAVING MULTIVARIATE MODEL PREDICTIONS LOCALLY ****')
            resFileName = resName.replace(' ', '_')
            fileName = f'{resFileName.upper()}_MULTIVARIATE_PREDICTIONS.csv'
            preds.to_csv(fileName)
        else:
            print('\n **** SAVING MULTIVARIATE MODEL PREDICTIONS LOCALLY ****')
            resFileName = resName.replace(' ', '_')
            fileName = f'{resFileName.upper()}_MULTIVARIATE_PREDICTIONS.csv'
            preds.to_csv(fileName)