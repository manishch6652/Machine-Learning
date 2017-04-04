import pandas as pd
import numpy as np
import math
import quandl #fiancial data python module
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime
import pickle #Serialization of python objects
style.use('ggplot')
df = quandl.get('WIKI/GOOGL') #google stock prices
df = df[['Adj. Open','Adj. Low','Adj. High','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close']*100;  #percent change in higher value
df['DL_PCT'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open']*100;  #Daily percent change 
#print(df.tail())
df = df[['Adj. Close','HL_PCT','DL_PCT','Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(-99999,inplace = True) #Fill NAN col values with -99999
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)# no of days to predict in near future
df['label'] = df[forecast_col].shift(-forecast_out) #forecasted value of stock
X = np.array(df.drop(['label'],1))
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
#Y = np.array(df['label'])
X = preprocessing.scale(X)
df.dropna(inplace = True)
Y = np.array(df['label'])
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size = 0.20)
clf = LinearRegression(n_jobs = -1)
clf.fit(X_train,Y_train)
with open('linearregression.pickle','wb') as f:
	pickle.dump(clf,f)
pickle_in = open('linearregression.pickle','rb')	
pickle = pickle.load(pickle_in)

accuracy = clf.score(X_test,Y_test)
forecast_set = clf.predict(X_lately) #set of forecast values of 30 days
print(forecast_set,accuracy,forecast_out)
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1) ]+ [i]


df['Adj. Close'].plot()
df['Forecast'].plot()

plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()