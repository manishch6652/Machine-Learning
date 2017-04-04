from statistics import mean
import numpy as np
import matplotlib.pyplot as plt 
from  matplotlib import style
style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6],dtype = np.float64)
ys = np.array([5,4,6,5,6,7],dtype = np.float64)
# calculation of slope
def best_fit_slope(xs,ys):
	m = (((mean(xs))*(mean(ys)))-((mean(xs*ys))))/(mean(xs)*mean(xs) - mean(xs*xs)) 
	return m
m =	best_fit_slope(xs,ys)	
print(m)
# calculation of y-intersect c = mean(y) - m*mean(x)
def intersct(m,xs,ys):
	c = mean(ys) - m*mean(xs)
	return c
c = intersct(m,xs,ys)
print(c)
#prediction of y for x
predict_x = 8
predict_y = (m*predict_x) + c
#print(predict_y) 	
line = [(m*x)+c for x in xs]
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color = 'g')
plt.plot(xs,line)
plt.show()



