
# coding: utf-8

# In[3]:

from numpy import *
import numpy as np


# In[4]:

x = np.array([0,1,2,3,4,5])


# In[5]:

y = np.array([0,0.8,0.1,0.9,-0.8,-1])


# In[6]:

from scipy.interpolate import *


# In[7]:

p1 = polyfit(x,y,1) #here we are estimating 1-D line to best fit in data


# In[8]:

print(p1)


# In[9]:

from matplotlib.pyplot import *


# In[10]:

#get_ipython().magic('matplotlib inline')


# In[11]:

p2= polyfit(x,y,2)#he we are estimating 2D or quardatic curve to best fit in data 
p3= polyfit(x,y,3)#he we are estimating 3degree  curve to best fit in data 
p4= polyfit(x,y,4)#he we are estimating 4-Degree curve to best fit in data 
plot(x,y,'o') 
plot(x,polyval(p1,x),'r-')
plot(x,polyval(p2,x),'b--')
plot(x,polyval(p3,x),'m')
plot(x,polyval(p4,x),'y-')


# In[15]:

yfit = p1[0]*x + p1[1]  
print(yfit)  #calculated val of y
print(y) #real val of y


# In[17]:

yresid = y - yfit
sresid = sum(pow(yresid,2))
stotal = len(y)*var(y)
mse = 1-sresid/stotal
print(mse) #mean square error


# In[18]:

from scipy.stats import *


# In[24]:

slope,intersect,r_value,p_value,std_err = linregress(x,y) #built-in linear regression function
print(pow(r_value,2))


# In[25]:

print(p_value)


# In[ ]:



