# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:59:47 2022

@author: river
"""

'''import pandas as pd
directory = "D:\\DEBI\\Uottawa\\Fundamentals-Applied DS\\project\\"
train_data  = pd.read_csv(directory+"X.csv")
target_data = pd.read_csv(directory+"Y.csv")
import datetime
# 2014-09-17   2019-11-24
target_data['6'] = target_data['6'].apply(lambda d: datetime.datetime.strptime(d, '%Y-%m-%d'))


import matplotlib.pyplot as plt
plt.plot(target_data['6'],target_data['4'],label = "Close")
# conver date string to Date object
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility

tf.random.set_seed(0)

import pandas as pd
import seaborn as sb
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

directory = "D:\\DEBI\\Uottawa\\Fundamentals-Applied DS\\project\\Bitcoin-price-prediction\\code\\"
dataFrame  = pd.read_csv(directory+"Final_Data2.csv")
dataFrame.drop('Unnamed: 0', axis=1, inplace=True)
print(dataFrame['timestamp'].unique())
# conver date string to Date object

import datetime
#import matplotlib.dates
dataFrame['timestamp'] = dataFrame['timestamp'].apply(lambda d: datetime.datetime.strptime(d, '%Y-%m-%d'))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, target = -1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(dataset[i + look_back, target])
	return np.array(dataX), np.array(dataY)

#df = dataFrame[["High","Low","Open","Volume","Marketcap","Close"]]
dataset = dataFrame[["Volume","Close"]].values
dataset = dataset.astype('float32')

# normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset= scaler.fit_transform(dataset)
X_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

y_t = y_scaler.fit_transform(dataset[:,-1].reshape(-1, 1))
dataset= X_scaler.fit_transform(dataset)
look_back = 1
X_lstm = dataset[:-look_back, :]
y_lstm = dataset[look_back:, -1]
#train_size = int(len(dataset) * 0.80)
#test_size = len(dataset) - train_size
#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1

#trainX, trainY = create_dataset(train, look_back)
#testX, testY = create_dataset(test, look_back)
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)

#trainX, testX ,trainY, testY = train[0:train_size,:], train[train_size:,:], test[0:train_size,:], test[train_size:,:]

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[1]))
'''

X_lstm = df.iloc[:-1, :].values
y_lstm = df.iloc[1:, -1].values
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
train_transformed = scaler.fit_transform(X_lstm.reshape(-1, 2))
target_transformed = scaler.fit_transform(y_lstm.reshape(-1, 1))

# split into train and test sets
train_size = int(len(train_transformed) * 0.80)
test_size = len(train_transformed) - train_size
X_train, X_test ,y_train, y_test = train_transformed[0:train_size,:], train_transformed[train_size:,:], target_transformed[0:train_size,:], target_transformed[train_size:,:]

print(len(X_train), len(X_test))
print(len(y_train), len(y_test))


'''
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = y_scaler.inverse_transform(trainPredict)
trainY = y_scaler.inverse_transform(trainY.reshape(-1, 1))
testPredict = y_scaler.inverse_transform(testPredict)
testY = y_scaler.inverse_transform(testY.reshape(-1, 1))



# calculate root mean squared error
print("LSTM with look_back = ",look_back)
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
predict_Train =trainPredict
predict = testPredict
print(f"mean_squared_error : {mean_squared_error(testY, predict)}")
print(f"mean_absolute_error : {mean_absolute_error(testY, predict)}")
print("----------------")
print(f"mean_squared_error Train : {mean_squared_error(trainY, predict_Train)}")
print(f"mean_absolute_error Train : {mean_absolute_error(trainY, predict_Train)}")
print("----------------")
print("Train R2 score:", r2_score(trainY, predict_Train))
print("Test R2 score:", r2_score(testY, predict))

'''
# shift train predictions for plotting
trainPredictPlot = dataset.copy()
trainPredictPlot[:, 1] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(X_scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()'''
# shift train predictions for plotting

def plot_pred_train_test(trainPredict ,testPredict,dataset =dataFrame[["timestamp","Close"]].values
                        , title = "Model eval",xLabel ="X",yLabel = "Y"):
    
    dataset = dataFrame[["timestamp","Close"]].values#np.concatenate(( X[:,-1], y[-1:]), axis=0)
    trainPredictPlot = np.empty((len(dataset), 1))
    trainPredictPlot[:] = np.nan
    
    trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty((len(dataset), 1))
    testPredictPlot[:] = np.nan
    
    testPredictPlot[len(trainPredict)+look_back:, :] = testPredict
    # plot baseline and predictions
    plt.plot(dataset[:,0],dataset[:,1], label = "actual")
    plt.plot(dataset[:,0],trainPredictPlot, label = "train prediction")
    plt.plot(dataset[:,0],testPredictPlot, label = "test prediction")
    plt.title(title)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend()
    plt.show()

plot_pred_train_test(trainPredict ,testPredict,title = "LSTM - training over (Volume,Close)",
                     xLabel ="Date", yLabel= "Close price")



#############################################################################################3

'''
dataset2 =dataFrame[["timestamp","Close"]].values
last_n = 5
x_axes_labels = dataset2[-last_n:,0]

x = np.arange(len(x_axes_labels))  # the label locations
fig, ax = plt.subplots()
ax.plot(x,dataset2[-last_n:,1], label = "actual")
ax.plot(x,testPredictPlot[-last_n:], label = "test prediction")
ax.set_title(f'LSTM Model prediction for last {last_n} days\n- training over close only')
ax.set_ylabel('Close')
ax.set_xlabel('date')
ax.set_xticks(x, x_axes_labels , rotation='vertical')
ax.legend()
fig.tight_layout()
plt.show()

 

# Date Jul 20, 2022
#["High","Low","Open","Volume","Marketcap","Close"]
#Open*	High	Low	Close**	Volume	Market Cap
#$23,393.19	$24,196.82	$23,009.95	$23,231.73	$42,932,549,127	$443,696,738,856
test_day_close = 23389.43#23231.73
test_input = np.array([test_day_close])
test_input = scaler.fit_transform(test_input[:].reshape(-1, 1))
test_input = np.reshape(test_input, (test_input.shape[0], 1, test_input.shape[1]))
pred= model.predict(test_input)

LSTM_day_pred = scaler.inverse_transform(pred)[0][0]
actual_day_value = 23164.63
x_axes_labels = ["Jul_202022_close","actual","LSTM","LinearR"]

x = np.arange(len(x_axes_labels))  # the label locations
fig, ax = plt.subplots()
b1 = ax.bar(0,test_day_close)
b2 = ax.bar(1,actual_day_value)
b3 = ax.bar(2,LSTM_day_pred)
b4 = ax.bar(3,lr_day_pred)
plt.bar_label(b1, fmt='%.2f')
plt.bar_label(b2, fmt='%.2f')
plt.bar_label(b3, fmt='%.2f')
plt.bar_label(b4, fmt='%.2f')

ax.set_ylim(23000,23400)
ax.set_title(f'Compare models results for predicting\n the close for Jul 21, 2022')
ax.set_ylabel('Close')
ax.set_xlabel('date')
ax.set_xticks(x, x_axes_labels , rotation='vertical')

ax.legend()
fig.tight_layout()
plt.show()
'''
