import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd


#2 = benign
#4 = malignant

df = pd.read_csv('breast-cancer-wisconsin.data.txt')

#replace all 16 ?
df.replace('?', -99999, inplace = True) #-99999 treats as outliners  or df.dropna() we can use
df.drop(['id'], 1, inplace = True)  #drop id column bcz not needed

#X features y labels

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])



X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()  #k neighbour classigfier
clf.fit(X_train,y_train)


accuracy = clf.score(X_test,y_test)
#print(accuracy)


example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,3,1,1,2,3,2,1]]) #we can add more also
example_measures = example_measures.reshape(len(example_measures),-1) #len(example_measure)=1 if there is on list 2 becasue 2 list not needed but doing because of warning
prediction = clf.predict(example_measures)
print(prediction)
