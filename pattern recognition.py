#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

c = [1,1,2,2,3,4]
d = [1,2,2,3,0,0]
    
a = [3,3,3]
b = [1,1,1]

plt.scatter(c,d,c='red')
plt.scatter(a,b,c='blue')
plt.show()


# In[2]:


import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

list=[[1,1,'A'],[1,2,'A'],[2,2,'A'],[2,3,'A'],[3,0,'A'],[4,0,'A'],[3,1,'B'],[3,1,'B'],[3,1,'B']]
df = pd.DataFrame(list,columns=['x','y','class'])
df1 = pd.DataFrame(list,columns=['x','y','class'])
y1=df1['class'].values
print(df)

label_encoder=preprocessing.LabelEncoder()
df['class']=label_encoder.fit_transform(df['class'])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)

x=df[['x','y']]
y=df['class']

svm_model = svm.SVC(kernel='linear')
svm_model.fit(x,y)

y_pred = svm_model.predict(x)
s=accuracy_score(y_pred,y)

print(s)

e=df[['x','y']].values
r=df['class'].values


# In[3]:


def draw(x,y,model):
    from mlxtend.plotting import plot_decision_regions
    import matplotlib.pyplot as plt
    plot_decision_regions(x,y,clf=model)
    plt.show()

draw(e,r,svm_model)

w=svm_model.coef_[0]
print(w)
print(svm_model.intercept_[0])


# In[4]:


from sklearn.linear_model import LogisticRegression
LRM = LogisticRegression(random_state=0,max_iter=200)
LRM.fit(x,y1)
y_pred = LRM.predict(x)
accuracy_score(y_pred,y1)


# In[12]:


svm_model = svm.SVC(kernel='poly',degree=3)
svm_model.fit(e,r)

y_pred = svm_model.predict(e)
s=accuracy_score(y_pred,r)

print(s)

e=df[['x','y']].values
r=df['class'].values
draw(e,r,svm_model)


# In[7]:


from sklearn.naive_bayes import GaussianNB
model_GNB = GaussianNB()
model_GNB.fit(e,r)
y_pred = model_GNB.predict(e)
print(accuracy_score(y_pred,r))
draw(e,r,model_GNB)

