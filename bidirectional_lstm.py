# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:07:34 2022

@author: Salih Eren
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM,Flatten
from keras.layers import ConvLSTM2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator
# load the dataset

import time

# get the start time
st = time.time()

dataframe = read_csv('C:\\Users\\Salih Eren\\.spyder-py3\\inflation.csv')
dataframe = dataframe[['Month', 'Value']]
print("Total data", len(dataframe))
dataframe['Month'] = pd.to_datetime(dataframe['Month'])
print(dataframe.dtypes)


dataframe.set_index('Month', inplace=True)

#plt.plot(dataframe['Value'])

scaler = MinMaxScaler(feature_range=(0, 1)) 
dataframe = scaler.fit_transform(dataframe)


train_size = int(len(dataframe) * 0.66)
test_size = len(dataframe) - train_size
train, test = dataframe[0:train_size,:], dataframe[train_size:len(dataframe),:]

def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        #print(i)
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])

    return np.array(x),np.array(y)

AAtrain_rmse=[]
AAtest_rmse=[]
AAtrain_mae=[]
AAtest_mae=[]
AAtrain_mse=[]
AAtest_mse=[]
i=0

while i<1:
    seq_size = 12  # Number of time steps to look back
    
    trainX, trainY = to_sequences(train, seq_size)
    testX, testY = to_sequences(test, seq_size)
    
    
    print("Shape of training set: {}".format(trainX.shape))
    print("Shape of test set: {}".format(testX.shape))
    from sklearn.metrics import mean_absolute_error as mean_absolute_error
    
    
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    from keras.layers import Bidirectional
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(None, seq_size)))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    
    model.fit(trainX, trainY, validation_data=(testX, testY),
              verbose=2, epochs=100)
    
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
    
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score using Root Mean Square Error For BiLSTM:: %.2f RMSE' % (trainScore))
    
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score using Root Mean Square Error For BiLSTM : %.2f RMSE' % (testScore))
    AAtrain_rmse.append(trainScore)
    AAtest_rmse.append(testScore)
    
    # calculate mean absolute error
    trainMaeScore = mean_absolute_error(trainY[0], trainPredict[:,0])
    print('Train Score using Mean Absolute Error for BiLSTM : %.2f MAE' % (trainMaeScore))
    
    testMaeScore = mean_absolute_error(testY[0], testPredict[:,0])
    print('Test Score using Mean Absolute Error for BiLSTM : %.2f MAE\n' % (testMaeScore))
    AAtrain_mae.append(trainMaeScore)
    AAtest_mae.append(testMaeScore)
    # calculate mean square error
    trainMseScore = mean_squared_error(trainY[0], trainPredict[:,0])
    #print('Train Score using Mean Square Error for (64)+Dense(32)+Dense(1) : %.2f MSE' % (trainMseScore))
    
    testMseScore = mean_squared_error(testY[0], testPredict[:,0])
    #print('Test Score using Mean Square Error for (64)+Dense(32)+Dense(1): %.2f MSE \n' % (testMseScore))
    AAtrain_mse.append(trainMseScore)
    AAtest_mse.append(testMseScore)
    i=i+1
    
trainPredictPlot = np.empty_like(dataframe)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataframe)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(seq_size*2)+1:len(dataframe)-1, :] = testPredict

MAE_sum=sum(AAtest_mae)
RMSE_sum=sum(AAtest_rmse)
MSE_sum=sum(AAtest_mse)
AAAtotal_mae=MAE_sum/i
AAAtotal_rmse=RMSE_sum/i
AAAtotal_mse=MSE_sum/i

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataframe))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)


plt.legend(['Base Data', 'Training data','Prediction data'])

et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')