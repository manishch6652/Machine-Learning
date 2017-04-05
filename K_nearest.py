import numpy as np 
from sklearn import preprocessing , cross_validation,neighbors
import pandas as pd 
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('NAN',-99999,inplace = True)
df.replace('?',-99999,inplace = True) #replace NAN with -99999
df.drop(['id'],1,inplace = True) #id is useless so dropping it
#print(df.head())
#We have to drop class column bcz that what output we expect for
x = np.array(df.drop(['class'],1))
y = np.array(df['class'])
x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size = 0.2)
#classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
accuracy = clf.score(x_test,y_test)
print(accuracy)
#predication 
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)
predication = clf.predict(example_measures)
print(predication)