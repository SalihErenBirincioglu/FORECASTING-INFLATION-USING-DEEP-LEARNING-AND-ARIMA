# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 22:09:19 2022

@author: Salih Eren
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
singleLSTM_RMSE_seq12=[[50,100,150,250],
                       [2.448798667432965,2.4599045762741447,2.4979141889383407,2.487144618124067]]

singleLSTM_MAE_seq12=[[50,100,150,250],
                       [1.4320193609282046,1.4267226852383361,1.4829097255700625,1.4657863438339942]]


stackedLSTM_RMSE_seq12=[[50,100,150,250],
                       [2.5784205715239734,2.5297450640763124,2.6006814175443087,2.5677215932058237]]

stackedLSTM_MAE_seq12=[[50,100,150,250],
                       [1.5150008965123547,1.4521478008283701,1.5406085351536947,1.5108111605795222]]


NN_RMSE_seq12=[[50,100,150,250],
                       [2.4705915710804764,2.4690464130275855,2.461960354394082,2.496969621443079]]
NN_MAE_seq12=[[50,100,150,250],
                       [1.4319284975808246,1.4293878940640241,1.6083891408668458,1.5220599790639588]]




BiDirectionalLSTM_RMSE_seq12=[[50,100,150,250],
                       [2.4310360280706056,2.4609324844545775,2.485309471518292,2.489004334632317]]
BiDirectionalLSTM_MAE_seq12=[[50,100,150,250],
                       [1.5616833693750092,1.5990640098868258,1.461950573980977,1.4692059854660646]]





plt.figure(figsize=(16,14))
plt.subplot(411)
plt.plot(singleLSTM_RMSE_seq12[0],singleLSTM_RMSE_seq12[1],label='LSTM RMSE', color='blue')
plt.legend(loc='upper left')
plt.plot(stackedLSTM_RMSE_seq12[0],stackedLSTM_RMSE_seq12[1],label='Stacked LSTM RMSE', color='red')
plt.legend(loc='upper left')
plt.plot(BiDirectionalLSTM_RMSE_seq12[0],BiDirectionalLSTM_RMSE_seq12[1],label='BiLSTM RMSE', color='green')
plt.legend(loc='lower left')
plt.plot(NN_RMSE_seq12[0],NN_RMSE_seq12[1],label='NN RMSE', color='orange')
plt.legend(loc='upper left')
plt.xlabel('Number of Compilations')
plt.ylabel('Error Rate')
plt.figure(figsize=(16,14))
plt.subplot(412)
plt.plot(singleLSTM_MAE_seq12[0],singleLSTM_MAE_seq12[1],label='LSTM MAE', color='blue')
plt.legend(loc='upper left')
plt.plot(stackedLSTM_MAE_seq12[0],stackedLSTM_MAE_seq12[1],label='Stacked LSTM MAE', color='red')
plt.legend(loc='upper left')
plt.plot(BiDirectionalLSTM_MAE_seq12[0],BiDirectionalLSTM_MAE_seq12[1],label='BiLSTM MAE', color='green')
plt.legend(loc='lower left')
plt.plot(NN_MAE_seq12[0],NN_MAE_seq12[1],label='NN MAE', color='orange')
plt.legend(loc='upper left')
plt.xlabel('Number of Compilations')
plt.ylabel('Error Rate')



singleLSTM_RMSE_seq36=[[50,100,150,250],
                       [2.212035819215133,2.229541649032222,2.243392321806433,2.241046005132085]]

singleLSTM_MAE_seq36=[[50,100,150,250],
                       [1.2344183087259026,1.2421600062310687,1.2502858442645193,1.2585376670147237]]


stackedLSTM_RMSE_seq36=[[50,100,150,250],
                       [2.586763736978193,2.586701411633743,2.5590750174576016,2.5989483193450402]]

stackedLSTM_MAE_seq36=[[50,100,150,250],
                       [1.4917476496704711,1.48479932417171031,1.4621526071520048,1.505845348371877]]


NN_RMSE_seq36=[[50,100,150,250],
                       [2.228040256660943,2.4929308550028737,2.4572966242850462,2.4993224847241446]]
NN_MAE_seq36=[[50,100,150,250],
                       [1.2610682684428183,1.6256404083723743,1.607792023429466,1.6417891246399878]]


BiDirectionalLSTM_RMSE_seq36=[[50,100,150,250],
                       [2.5061931536037334,2.5363141347757012,2.4965662326864786,2.505562748714262]]
BiDirectionalLSTM_MAE_seq36=[[50,100,150,250],
                       [1.4590299058333716,1.4869504821245705,1.477339559141954,1.4897312362095296]]

plt.figure(figsize=(16,14))
plt.subplot(413)
plt.plot(singleLSTM_RMSE_seq36[0],singleLSTM_RMSE_seq36[1],label='LSTM RMSE', color='blue')
plt.legend(loc='lower left')
plt.plot(stackedLSTM_RMSE_seq36[0],stackedLSTM_RMSE_seq36[1],label='Stacked LSTM RMSE', color='red')
plt.legend(loc='lower left')
plt.plot(BiDirectionalLSTM_RMSE_seq36[0],BiDirectionalLSTM_RMSE_seq36[1],label='BiLSTM RMSE', color='green')
plt.legend(loc='lower left')
plt.plot(NN_RMSE_seq36[0],NN_RMSE_seq36[1],label='NN RMSE', color='orange')
plt.legend(loc='lower left')
plt.xlabel('Number of Compilations')
plt.ylabel('Error Rate')
plt.figure(figsize=(16,14))
plt.subplot(414)
plt.plot(singleLSTM_MAE_seq36[0],singleLSTM_MAE_seq36[1],label='LSTM MAE', color='blue')
plt.legend(loc='lower left')
plt.plot(stackedLSTM_MAE_seq36[0],stackedLSTM_MAE_seq36[1],label='Stacked LSTM MAE', color='red')
plt.legend(loc='lower left')
plt.plot(BiDirectionalLSTM_MAE_seq36[0],BiDirectionalLSTM_MAE_seq36[1],label='BiLSTM MAE', color='green')
plt.legend(loc='lower left')
plt.plot(NN_MAE_seq36[0],NN_MAE_seq36[1],label='NN MAE', color='orange')
plt.legend(loc='lower left')
plt.xlabel('Number of Compilations')
plt.ylabel('Error Rate')



"""

singleLSTM_RMSE_seq60=[[50,100,150,250],
                       [2.477767493196673,2.482685893995834,2.4872795332673405,2.487707121950724]]

singleLSTM_MAE_seq60=[[50,100,150,250],
                       [1.454499467786604 ,1.4707977534393777,1.472666550846448,1.474308450364213]]


stackedLSTM_RMSE_seq60=[[50,100,150,250],
                       [2.8809482572218403,2.8968909324158627,2.8815473695778198,2.8838132939372967]]

stackedLSTM_MAE_seq60=[[50,100,150,250],
                       [1.7725933385195987,1.7799797004995008,1.7633768656459032,1.7587514047160904]]


NN_RMSE_seq60=[[50,100,150,250],
                       [2.8815454890325403,2.9362903262190083,2.9990801541884675,2.92010167655751]]
NN_MAE_seq60=[[50,100,150,250],
                       [1.9022526304932759,1.9160325455146452,1.9921184545045802,1.9417584738506861]]


BiDirectionalLSTM_RMSE_seq60=[[50,100,150,250],
                       [3.050150115141818,3.074655844694486,3.070088829109188,3.0703039665754592]]
BiDirectionalLSTM_MAE_seq60=[[50,100,150,250],
                       [1.9629395637471216,1.9595931937280913,1.9564270590505302,1.940754824210967]]

plt.figure(figsize=(16,14))
plt.subplot(413)
plt.plot(singleLSTM_RMSE_seq60[0],singleLSTM_RMSE_seq60[1],label='LSTM RMSE', color='blue')
plt.legend(loc='lower left')
plt.plot(stackedLSTM_RMSE_seq60[0],stackedLSTM_RMSE_seq60[1],label='Stacked LSTM RMSE', color='red')
plt.legend(loc='lower left')
plt.plot(BiDirectionalLSTM_RMSE_seq60[0],BiDirectionalLSTM_RMSE_seq60[1],label='BiLSTM RMSE', color='green')
plt.legend(loc='lower left')
plt.plot(NN_RMSE_seq60[0],NN_RMSE_seq60[1],label='NN RMSE', color='orange')
plt.legend(loc='lower left')
plt.xlabel('Number of Compilations')
plt.ylabel('Error Rate')
plt.figure(figsize=(16,14))
plt.subplot(414)
plt.plot(singleLSTM_MAE_seq60[0],singleLSTM_MAE_seq60[1],label='LSTM MAE', color='blue')
plt.legend(loc='lower left')
plt.plot(stackedLSTM_MAE_seq60[0],stackedLSTM_MAE_seq60[1],label='Stacked LSTM MAE', color='red')
plt.legend(loc='lower left')
plt.plot(BiDirectionalLSTM_MAE_seq60[0],BiDirectionalLSTM_MAE_seq60[1],label='BiLSTM MAE', color='green')
plt.legend(loc='lower left')
plt.plot(NN_MAE_seq60[0],NN_MAE_seq60[1],label='NN MAE', color='orange')
plt.legend(loc='lower left')
plt.xlabel('Number of Compilations')
plt.ylabel('Error Rate')









plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(singleLSTM_RMSE_seq12[0],singleLSTM_RMSE_seq12[1],label='LSTM RMSE Seq_12', color='blue')
plt.legend(loc='upper left')
plt.plot(singleLSTM_RMSE_seq36[0],singleLSTM_RMSE_seq36[1],label='LSTM RMSE Seq_36', color='red')
plt.legend(loc='upper left')
plt.plot(singleLSTM_RMSE_seq60[0],singleLSTM_RMSE_seq60[1],label='LSTM RMSE Seq_60', color='orange')
plt.legend(loc='upper left')

plt.figure(figsize=(12,8))
plt.subplot(412)
plt.plot(singleLSTM_MAE_seq12[0],singleLSTM_MAE_seq12[1],label='LSTM MAE Seq_12', color='blue')
plt.legend(loc='upper left')
plt.plot(singleLSTM_MAE_seq36[0],singleLSTM_MAE_seq36[1],label='LSTM MAE Seq_36', color='red')
plt.legend(loc='upper left')
plt.plot(singleLSTM_MAE_seq60[0],singleLSTM_MAE_seq60[1],label='LSTM MAE Seq_60', color='orange')
plt.legend(loc='upper left')





"""

plt.figure(figsize=(12,8))
plt.subplot(413)
plt.plot(NN_RMSE_seq12[0],NN_RMSE_seq12[1],label='NN RMSE Seq_12', color='blue')
plt.legend(loc='upper left')
plt.plot(NN_RMSE_seq36[0],NN_RMSE_seq36[1],label='NN RMSE Seq_36', color='red')
plt.legend(loc='upper left')
plt.plot(NN_RMSE_seq60[0],NN_RMSE_seq60[1],label='NN RMSE Seq_60', color='orange')
plt.legend(loc='upper left')

plt.figure(figsize=(12,8))
plt.subplot(414)
plt.plot(NN_MAE_seq12[0],NN_MAE_seq12[1],label='NN MAE Seq_12', color='blue')
plt.legend(loc='upper left')
plt.plot(NN_MAE_seq36[0],NN_MAE_seq36[1],label='NN MAE Seq_36', color='red')
plt.legend(loc='upper left')
plt.plot(NN_MAE_seq60[0],NN_MAE_seq60[1],label='NN MAE Seq_60', color='orange')
plt.legend(loc='upper left')

"""

