#!/usr/bin/env python
# coding: utf-8

# In[1]:


#KNN

import pandas as pd
tshirt=pd.read_csv("C:/Amrita/Machine Learning/tshirt.csv")
df=pd.DataFrame(tshirt)
import numpy as np
h=int(input("Enter Height:"))
w=int(input("Enter Weight:"))
k=int(input("Enter value of K:"))
c=df['Height']-h
d=df['Weight']-w
x=(c*c)
y=(d*d)
xy=np.sqrt(x+y)
df.insert(3,"Distance",xy)
df2=df.sort_values(by=['Distance'])
s=df2.head(k)
print(s)
z=s['T-shirt size'].value_counts()
if (len(s.index)==2):
    if (z[0]>z[1]):
        print("The T-shirt size is:",z.index[0])
    else:
        print("The T-shirt size is:",z.index[1])
else:
    print("The T-shirt size is:",z.index[0])
del df['Distance']


# In[5]:


#K-means
x=[0,0.5,0,2,2.5,6,7]
y=[0,0,2,2,8,3,3]
c1=[0,0]
c2=[2,2]
c3=[7,3]
count=0

while True:
    d1=[]
    d2=[]
    d3=[]
    pb=[]
    ncx1=[]
    ncy1=[]
    ncx2=[]
    ncy2=[]
    ncx3=[]
    ncy3=[]
    c4=[c1[0],c1[1]]
    c5=[c2[0],c2[1]]
    c6=[c3[0],c3[1]]
    
    for j in range(0,len(x),1):
        d1.append(abs(x[j]-c1[0])+abs(y[j]-c1[1]))
    for i in range(0,len(x),1):
        d2.append(abs(x[i]-c2[0])+abs(y[i]-c2[1]))
    for k in range(0,len(x),1):
        d3.append(abs(x[k]-c3[0])+abs(y[k]-c3[1]))
    for l in range(0,len(x),1):
        if(min(d1[l],d2[l],d3[l])==d1[l]):
                pb.append('c1')
        elif(min(d1[l],d2[l],d3[l])==d2[l]):
                pb.append('c2')
        elif(min(d1[l],d2[l],d3[l])==d3[l]):
                pb.append('c3')
    for m in range(0,len(x),1):
        if(pb[m]=='c1'):
            ncx1.append(x[m])
            ncy1.append(y[m])
        elif(pb[m]=='c2'):
            ncx2.append(x[m])
            ncy2.append(y[m])
        else:
            ncx3.append(x[m])
            ncy3.append(y[m])
        
    ncx1=sum(ncx1)/len(ncx1)
    ncy1=sum(ncy1)/len(ncy1)
    c1[0]=ncx1
    c1[1]=ncy1
    print(c1)
    ncx2=sum(ncx2)/len(ncx2)
    ncy2=sum(ncy2)/len(ncy2)
    c2[0]=ncx2
    c2[1]=ncy2
    print(c2)
    ncx3=sum(ncx3)/len(ncx3)
    ncy3=sum(ncy3)/len(ncy3)
    c3[0]=ncx3
    c3[1]=ncy3
    print(c3)
    count=count+1
    if (c4==c1 and c5==c2):
        if c6==c3:
            print('Number of iterations is:',+count)
            print('centroid 1:',c1)
            print('centroid 2:',c2)
            print('centroid 3:',c3)
            print('points belong to:',pb)
            break


# In[2]:


import pandas as pd
df1=pd.read_csv("C:/Users/senth/Downloads/Iris.csv")
df2=pd.read_csv("C:/Users/senth/Downloads/PlayTennisN.csv")
print(df1)
df1.info()


# In[3]:


df11=pd.read_csv("C:/Users/senth/Downloads/Iris.csv",usecols=['SepalLengthCm','PetalLengthCm'],nrows=5,skiprows=[1,4])


# In[5]:


print(df1['Species'].value_counts())
df1


# In[6]:


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()
df1['Species']= label_encoder.fit_transform(df1['Species'])
df1


# In[5]:


#Decision Tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
df1
features=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
x=df1[features]
y=df1['Species']
dtree = DecisionTreeClassifier()
dtree = dtree.fit(x, y)

plt.figure(figsize=(15,15))
tree.plot_tree(dtree, feature_names=features)
plt.show()



# In[4]:


print(df1['Species'].count())


# In[12]:


print(dtree.predict([[4,3.2,2,1.9]]))


# In[84]:


import pandas as pd
from sklearn import preprocessing
df=pd.read_csv("C:/Users/senth/Downloads/drug200.csv")
label_encoder=preprocessing.LabelEncoder()
df['Sex']=label_encoder.fit_transform(df['Sex'])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)
df['BP']=label_encoder.fit_transform(df['BP'])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)
df['Cholesterol']=label_encoder.fit_transform(df['Cholesterol'])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)
df['Drug']=label_encoder.fit_transform(df['Drug'])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)


# In[83]:


le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)


# In[85]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
features=['Age','Sex','BP','Cholesterol','Na_to_K']
x=df[features]
y=df['Drug']
dtree= DecisionTreeClassifier()
dtree= dtree.fit(x,y)

plt.figure(figsize=(10,10))
tree.plot_tree(dtree, feature_names=features)
plt.show()


# In[86]:


print(dtree.predict([[25,1,2,1,8.88]]))


# In[7]:


#K-Means
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()
df1=pd.read_csv("C:/Users/senth/Downloads/drug200.csv")
df2 = pd.DataFrame(df1)
df=df2[['Na_to_K','Age']]

kmeans = KMeans(n_clusters=5).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['Na_to_K'], df['Age'], c=kmeans.labels_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()


# In[39]:


from sklearn.neighbors import KNeighborsClassifier
x=df2.drop(['Sex','BP','Cholesterol','Drug'],axis='columns')
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, df1['Drug'])
print(knn.predict([[23,8.8]]))


# In[17]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv("C:/Users/senth/Downloads/Iris.csv")
df.info()
x=df.drop(['Id','Species'], axis='columns')
y=df['Species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=25)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train, y_train)
print(knn.predict(x_test))
print(y_test)


# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()
df1=pd.read_csv("C:/Users/senth/Downloads/Iris.csv")
print(df1.info())
df2 = pd.DataFrame(df1)
df=df2[['SepalWidthCm','PetalLengthCm']]

kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)


plt.figure(figsize=(10,10))
plt.scatter(df['SepalWidthCm'], df['PetalLengthCm'], c=kmeans.labels_,s=100)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red',s=100)
plt.show()


# In[2]:



#SVM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
df=pd.read_csv("C:/Users/senth/Downloads/Fish.csv")
label_encoder=preprocessing.LabelEncoder()
df['Species']=label_encoder.fit_transform(df['species'])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping)

x = df[['Weight','Length1']]
y = df['Species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=25)
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(y_pred)
print(y_test)
print(df.info())



# In[19]:


from sklearn import datasets
from sklearn.svm
iris=datasets.load_iris()
iris.feature_names
X=iris["data"][:,(2,3)]
Y=iris["target"]
svm_cf=svc(kernel="linear")
svm_cf.fit(X,Y)
svm_cf.predict([[2.4,3.1]])


# In[20]:


print(df)


# In[ ]:




