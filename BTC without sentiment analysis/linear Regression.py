
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

import matplotlib.pyplot as plt

ax = plt.gca()

dataFrame.plot(kind='line',x='timestamp',y='Open',ax=ax)
dataFrame.plot(kind='line',x='timestamp',y='Close', ax=ax)
dataFrame.plot(kind='line',x='timestamp',y='High',ax=ax)
dataFrame.plot(kind='line',x='timestamp',y='Low', ax=ax)

ax.set_title('BTC price')
ax.set_ylabel('price')
plt.show()


'''
plt.plot(dataFrame.timestamp,dataFrame.Open,label = "open")
plt.plot(dataFrame.timestamp,dataFrame.Close,label = "Close")
plt.plot(dataFrame.timestamp,dataFrame.High,label = "High")
plt.plot(dataFrame.timestamp,dataFrame.Low,label = "Low" )
plt.legend()
plt.title("BTC prices")
plt.ylabel("price")
plt.xlabel("Date")
plt.show()

'''
ax = plt.gca()

dataFrame.plot(kind='line',x='timestamp',y='Volume',ax=ax)

ax.set_title('BTC volume')
ax.set_ylabel('volume')
plt.show()



'''
plt.plot(dataFrame.Date,dataFrame.Volume,label = "Volume" )
plt.plot(dataFrame.Date,dataFrame.Marketcap,label = "Marketcap" )
plt.legend()
plt.title("BTC volume and marketcap")
plt.ylabel("price")
plt.xlabel("Date")
plt.show()
'''
import seaborn as sns
'''
print(dataFrame.corr())
  
# plotting correlation heatmap
dataplot = sb.heatmap(dataFrame.corr(), cmap="YlGnBu", annot=True)
  
# displaying heatmap
plt.show()
'''
#
# Correlation between different variables
#
corr = dataFrame.corr()
#
# Set up the matplotlib plot configuration
#
f, ax = plt.subplots(figsize=(12, 10))
#
# Generate a mask for upper traingle
#
mask = np.triu(np.ones_like(corr, dtype=bool))
#
# Configure a custom diverging colormap
#
cmap = sns.diverging_palette(230, 20, as_cmap=True)
#
# Draw the heatmap
#
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)

# prints data that will be plotted
# columns shown here are selected by corr() since
# they are ideal for the plot




from sklearn.model_selection import train_test_split

df = dataFrame[["High","Low","Open","Volume","Marketcap","Close"]]
print(df.corr())
  
# plotting correlation heatmap
dataplot = sb.heatmap(df.corr(), cmap="YlGnBu", annot=True)
  
# displaying heatmap
plt.show()
#df = dataFrame[["Open","Close"]]
X = df.iloc[:-1, :].values
y = df.iloc[1:, -1].values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_t = sc_X.fit_transform(X)
y_t = sc_y.fit_transform(y.reshape(-1, 1))


'''
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_t ,y_t)
print(regressor.score(X_t, y_t))

plt.plot(dataFrame.Date, regressor.predict(X_t),label= "predicted",c= "r")
plt.plot(dataFrame.Date, y_t, label ="actual", color = 'green')
plt.title('SVR(kernel = rbf) open vs close')
plt.ylabel("Close")
plt.xlabel("Date")
plt.legend()
plt.show()
'''

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
df = dataFrame[["Volume","Close"]]
#df = dataFrame[["Open","Close"]]
X = df.iloc[:-1, :].values
y = df.iloc[1:, -1].values

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
train_transformed = X#scaler.fit_transform(X)
target_transformed = y.reshape(-1, 1)#scaler.fit_transform(y.reshape(-1, 1))

# split into train and test sets
train_size = int(len(train_transformed) * 0.80)
test_size = len(train_transformed) - train_size
X_train, X_test ,y_train, y_test = train_transformed[0:train_size,:], train_transformed[train_size:,:], target_transformed[0:train_size,:], target_transformed[train_size:,:]
from sklearn.model_selection import train_test_split

X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(len(X_train), len(X_test))
print(len(y_train), len(y_test))

reg = LinearRegression(normalize = True)
reg.fit(X_train,y_train)
predict_Train = reg.predict(X_train)
predict = reg.predict(X_test)

print("LinearRegression")
trainScore = np.sqrt(mean_squared_error(y_train, predict_Train))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(y_test, predict))
print('Test Score: %.2f RMSE' % (testScore))

print(f"test mean_squared_error : {mean_squared_error(y_test, predict)}")
#mean_squared_error : 157020.94669562386
print(f"test mean_absolute_error : {mean_absolute_error(y_test, predict)}")

print("----------------")

print(f"Train mean_squared_error : {mean_squared_error(y_train, predict_Train)}")
#mean_squared_error : 157020.94669562386
print(f"Train mean_absolute_error : {mean_absolute_error(y_train, predict_Train)}")
print("----------------")
print("Train R2 score:", r2_score(y_train, predict_Train))
print("Test R2 score:", r2_score(y_test, predict))


from LSTM import plot_pred_train_test

plot_pred_train_test(predict_Train.reshape(-1,1) ,predict.reshape(-1,1) ,title = "LinearRegression - training over (Volume,Close)",
                     xLabel ="Date", yLabel= "Close price")




#############################################################################################3

from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X)

#plt.scatter(X_pca[:,0],X_pca[:,1], color = 'red')
#X = sc_y.inverse_transform(X)
#y = sc_y.inverse_transform(y)
#plt.scatter(X_pca[:,0], X_pca[:,1])

plt.plot(dataFrame.Date[-2:], reg.predict(X[-2:]), label ="predicted",color = 'red')
plt.plot(dataFrame.Date[-2:], y[-2:], label ="actual", color = 'green')
plt.title('LinearRegression past 30 days \n["High","Low","Open","Volume","Marketcap","Close"] vs close')
plt.ylabel("Close")
plt.xlabel("date")
plt.legend()
plt.show()

last_n = 5
x_axes_labels = dataFrame.Date[-last_n:]

x = np.arange(len(x_axes_labels))  # the label locations
fig, ax = plt.subplots()
ax.plot(x, reg.predict(X[-last_n:]), label ="predicted",color = 'red')
ax.plot(x, y[-last_n:], label ="actual", color = 'green')
ax.set_title(f'LinearRegression last {last_n} days \n["High","Low","Open","Volume","Marketcap","Close"] vs close')
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
test_input_reg = np.array([24196.82, 23009.95, 23393.19, 42932549127,443696738856,23231.73])
print(reg.predict(test_input_reg.reshape(-1,6)))
lr_day_pred = reg.predict(test_input_reg.reshape(-1,6))[0]
