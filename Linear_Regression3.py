import pandas as pd
import numpy as np
import math
import quandl #fiancial data python module
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
df = quandl.get('WIKI/GOOGL') #google stock prices
df = df[['Adj. Open','Adj. Low','Adj. High','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close']*100;  #percent change in higher value
df['DL_PCT'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open']*100;  #Daily percent change 
#print(df.tail())
df = df[['Adj. Close','HL_PCT','DL_PCT','Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(-99999,inplace = True) #Fill NAN col values with -99999
forecast_out = int(math.ceil(0.01*len(df)))
prin(forecast_out)# no of days to predict in near future
df['label'] = df[forecast_col].shift(-forecast_out) #forecasted value of stock
df.dropna(inplace = True)
X = np.array(df.drop(['label'],1))
Y = np.array(df['label'])
X = preprocessing.scale(X)
Y = np.array(df['label'])
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size = 0.20)
clf = LinearRegression()
clf.fit(X_train,Y_train)
accuracy = clf.score(X_test,Y_test)
print(accuracy)