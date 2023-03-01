
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('C:\\Users\\Salih Eren\\.spyder-py3\\inflation1.csv')
df = df[['Month', 'Value']]

#print("Total data", len(df))
df['Month'] = pd.to_datetime(df['Month'])
print(df.dtypes)


df.set_index('Month', inplace=True)

import time

# get the start time
st = time.time()

#Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(df)
print("pvalue = ", pvalue, " if above 0.05, data is not stationary")

"""
#Extract and plot trend, seasonal and residuals. 
from statsmodels.tsa.seasonal import seasonal_decompose 
decomposed = seasonal_decompose(df['Value'],  
                            model ='additive')
"""
"""
from pmdarima.arima import auto_arima
#Autoarima gives the best model suited for the data
# p - number of autoregressive terms (AR)
# q - Number of moving avergae terms (MA)
# d - number of non-seasonal differences
#p, d, q represent non-seasonal components
#P, D, Q represent seasonal components
arima_model = auto_arima(df['Value'], start_p = 1, d=1, start_q = 1, 
                          max_p = 5, max_q = 5, max_d=5, m = 12, 
                          start_P = 0, D=1, start_Q=0, max_P=5, max_D=5, max_Q=5,
                          seasonal = True, 
                          trace = True, 
                          error_action ='ignore',   
                          suppress_warnings = True,  
                          stepwise = True, n_fits=50)           
  
# To print the summary 
print(arima_model.summary() ) #Note down the Model and details.
# Model: SARIMAX(1, 1, 0)x(5, 1, 0, 12) SARIMAX(2, 1, 1)x(5, 1, [], 12)

#SARIMAX(3, 1, 0)x(5, 1, 0, 12)
"""

# SARIMAX(1, 1, 1)x(5, 1, [], 12) ->>inflation1 için
# SARIMAX(1, 1, 0)x(5, 1, 0, 12) ->>inflation2 için
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mean_absolute_error


#Split data into train and test
size = int(len(df) * 0.66)
X_train, X_test = df[0:size], df[size:len(df)]

#from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX 

model = SARIMAX(X_train['Value'],  
                order = (1, 1, 0),  
                seasonal_order =(5, 1, 0, 12)) 
      
result = model.fit() 
result.summary() 

#Train prediction
start_index = 0
end_index = len(X_train)-1
train_prediction = result.predict(start_index, end_index) 
    
#Prediction
start_index = len(X_train)
end_index = len(df)-1
prediction = result.predict(start_index, end_index).rename('Predicted Value') 
    
  

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(X_train, train_prediction))
#print('Train Score for SARIMAX: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(X_test, prediction))
print('Test Error for SARIMAX: %.2f RMSE' % (testScore))

    
# calculate mean absolute error
trainMaeScore = mean_absolute_error(X_train, train_prediction)
#print('Train Score using Mean Absolute Error for SARIMAX : %.2f MAE' % (trainMaeScore))

testMaeScore = mean_absolute_error(X_test, prediction)
print('Test Error using Mean Absolute Error for SARIMAX : %.2f MAE\n' % (testMaeScore))
    
  
    
from sklearn.metrics import r2_score
score = r2_score(X_test, prediction)
print("R2 score is: ", score)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

plt.plot(df, label='Original data', color='blue')
plt.plot(X_train, label='Training', color='black')
plt.plot(prediction, label='Prediction', color='red')
plt.legend(loc='lower left')
plt.xlabel('Years')
plt.ylabel('CPI Values')
plt.show()