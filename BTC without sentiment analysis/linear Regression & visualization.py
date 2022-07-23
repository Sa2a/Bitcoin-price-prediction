
import pandas as pd
import seaborn as sb

directory = "D:\\DEBI\\Uottawa\\Fundamentals-Applied DS\\project\\"
dataFrame  = pd.read_csv(directory+"coin_Bitcoin.csv")

print(dataFrame['Date'].unique())
# conver date string to Date object
import datetime
#import matplotlib.dates
dataFrame['Date'] = dataFrame['Date'].apply(lambda d: datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))

import matplotlib.pyplot as plt
plt.plot(dataFrame.Date,dataFrame.Open,label = "open")
plt.plot(dataFrame.Date,dataFrame.Close,label = "Close")
plt.plot(dataFrame.Date,dataFrame.High,label = "High")
plt.plot(dataFrame.Date,dataFrame.Low,label = "Low" )
plt.legend()
plt.title("BTC prices")
plt.ylabel("price")
plt.xlabel("Date")
plt.show()

plt.plot(dataFrame.Date,dataFrame.Volume,label = "Volume" )
plt.plot(dataFrame.Date,dataFrame.Marketcap,label = "Marketcap" )
plt.legend()
plt.title("BTC volume and marketcap")
plt.ylabel("price")
plt.xlabel("Date")
plt.show()

# prints data that will be plotted
# columns shown here are selected by corr() since
# they are ideal for the plot



import numpy as np
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


from sklearn.linear_model import LinearRegression
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
reg = LinearRegression(normalize=True)
reg.fit(X,y)

print(reg.score(X,y))

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
