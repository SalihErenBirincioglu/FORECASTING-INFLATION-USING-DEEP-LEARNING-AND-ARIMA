# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
import tensorflow as tf
tf.version.VERSION

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import time

# get the start time
st = time.time()

# load the dataset
dataset = read_csv('C:\\Users\\Salih Eren\\.spyder-py3\\inflation.csv')
dataset = dataset[['Month', 'Value']]
print("Total data", len(dataset))
dataset['Month'] = pd.to_datetime(dataset['Month'])

print(dataset.dtypes)


dataset['Month'] = pd.to_datetime(dataset['Month'])
print(dataset.dtypes)

dataset.set_index('Month', inplace=True) 

#plt.plot(dataset)



scaler = MinMaxScaler(feature_range=(0, 1)) 
dataset = scaler.fit_transform(dataset)

#split dataset
train_size = int(len(dataset) * 0.66)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]





seq_size = 12 # Number of time steps to look back 


#TimeseriesGenerator to organize training data into the right format
from keras.preprocessing.sequence import TimeseriesGenerator # Generates batches 
batch_size = 1
train_generator = TimeseriesGenerator(train.reshape(-1), train.reshape(-1), length=seq_size, batch_size=batch_size)
print("Total number of samples in the original training data = ", len(train)) 
print("Total number of samples in the generated data = ", len(train_generator)) 



from sklearn.metrics import mean_absolute_error as mean_absolute_error
# print a couple of samples... 
#x, y = train_generator[0]
AAtrain_rmse=[]
AAtest_rmse=[]
AAtrain_mae=[]
AAtest_mae=[]
AAtrain_mse=[]
AAtest_mse=[]
i=0
while i<1:
    #Generate validation data
    validation_generator = TimeseriesGenerator(test.reshape(-1), test.reshape(-1), length=seq_size, batch_size=batch_size)
    print("Total number of samples in the validation data = ", len(validation_generator))
    #create and fit model
    model = Sequential()
    model.add(Dense(32, input_dim=seq_size, activation='relu')) #12
    model.add(Dense(16, activation='relu'))  #8
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    print(model.summary()) 
    
  
    model.fit_generator(generator=train_generator, verbose=2, epochs=100, validation_data=validation_generator)
    
    #make predictions
    
    trainPredict = model.predict(train_generator)
    testPredict = model.predict(validation_generator)
    
    """
    loss = history.history['loss']
    val_loss =history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    """
    
    #invert the transformation.
   
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY_inverse = scaler.inverse_transform(train)
    testPredict = scaler.inverse_transform(testPredict)
    testY_inverse = scaler.inverse_transform(test)
    
    # calculate root mean squared error
    trainRmseScore = math.sqrt(mean_squared_error(trainY_inverse[seq_size:], trainPredict[:,0]))
    #print('Train Score using Root Mean Square Error For NN : %.2f RMSE' % (trainRmseScore))
    
    testRmseScore = math.sqrt(mean_squared_error(testY_inverse[seq_size:], testPredict[:,0]))
    print('Test Error using Root Mean Square Error For NN : %.2f RMSE \n' % (testRmseScore))
    AAtrain_rmse.append(trainRmseScore)
    AAtest_rmse.append(testRmseScore)
    
    # calculate mean absolute error
    trainMaeScore = mean_absolute_error(trainY_inverse[seq_size:], trainPredict[:,0])
    #print('Train Score using Mean Absolute Error for NN : %.2f MAE' % (trainMaeScore))
    testMaeScore = mean_absolute_error(testY_inverse[seq_size:], testPredict[:,0])
    print('Test Error using Mean Absolute Error for NN : %.2f MAE\n' % (testMaeScore))
    AAtrain_mae.append(trainMaeScore)
    AAtest_mae.append(testMaeScore)
    
    
    # calculate mean square error
    """
    trainMseScore = mean_squared_error(trainY_inverse[seq_size:], trainPredict[:,0])
    #print('Train Score using Mean Square Error for (64)+Dense(32)+Dense(1) : %.2f MSE' % (trainMseScore))
    
    testMseScore = mean_squared_error(testY_inverse[seq_size:], testPredict[:,0])
    #print('Test Score using Mean Square Error for (64)+Dense(32)+Dense(1): %.2f MSE \n' % (testMseScore))
    AAtrain_mse.append(trainMseScore)
    AAtest_mse.append(testMseScore)
    """
    i=i+1
    
#we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(seq_size*2)-1:len(dataset)-1, :] = testPredict
testPredictPlot[len(train)+(seq_size)-1:len(dataset)-1, :] = testPredict

 
from sklearn.metrics import r2_score
score=r2_score(testY_inverse[seq_size:],testPredict[:,0])
print("R2 score is: ", score)

MAE_sum=sum(AAtest_mae)
RMSE_sum=sum(AAtest_rmse)
MSE_sum=sum(AAtest_mse)
AAAtotal_mae=MAE_sum/i
AAAtotal_rmse=RMSE_sum/i

#AAAtotal_mse=MSE_sum/5

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)

plt.legend(['Base Data', 'Training data','Prediction data'])


et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')