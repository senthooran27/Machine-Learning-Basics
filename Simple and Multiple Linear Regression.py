#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
model = LinearRegression()

print("simple linear regression:")
x = np.array([2,8,11,10,8,4,2,2,9,8,4,11,12,2,4,4,20,1,10,15,15,16,17,6,5]).reshape((-1, 1))
y = np.array([9.95,24.45,31.75,35.00,25.02,16.86,14.38,9.60,24.35,27.50,17.08,37.00,41.95,11.66,21.65,17.89,69.00,10.30,34.93,46.59,44.88,54.12,56.63,22.13,21.15]
)
model.fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

print("\nmultiple linear regression:")
x1= [[2, 50],
     [8,110],
     [11,120],
     [10,550],
     [8,295],
     [4,200],
     [2,375],
     [2,52],
     [9,100],
     [8,300],
     [4,412],
     [11,400],
     [12,500],
     [2,360],
     [4,205],
     [4,400],
     [20,600],
     [1,585],
     [10,540],
     [15,250],
     [15,290],
     [16,510],
     [17,590],
     [6,100],
     [5,400]]
x2=[50,110,120,550,295,200,375,52,100,300,412,400,500,360,205,400,600,585,540,250,290,510,590,100,400]
y1 =[9.95,24.45,31.75,35.00,25.02,16.86,14.38,9.60,24.35,27.50,17.08,37.00,41.95,11.66,21.65,17.89,69.00,10.30,34.93,46.59,44.88,54.12,56.63,22.13,21.15]
x11, y11 = np.array(x1), np.array(y1)

model = LinearRegression().fit(x11, y11)
r_sq = model.score(x11, y11)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"coefficients: {model.coef_}")


# In[4]:





# In[ ]:




