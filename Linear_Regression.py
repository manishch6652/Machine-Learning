import pandas as pd
import math
import quandl #fiancial data python module
df = quandl.get('WIKI/GOOGL') #google stock prices
df = df[['Adj. Open','Adj. Low','Adj. High','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close']*100;  #percent change in higher value
df['DL_PCT'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open']*100;  #Daily percent change 
print(df.tail())
