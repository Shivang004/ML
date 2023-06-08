#!/usr/bin/env python
# coding: utf-8

# In[205]:


import pandas as pd


# In[206]:


data=pd.read_csv("car.csv")


# In[207]:


data.head()


# In[208]:


pd.set_option('display.max_rows',10)


# In[209]:


data


# In[210]:


data.describe()


# In[211]:


data=data.dropna(axis=0)


# In[212]:


y=data.Price


# In[213]:


x=data[['Brand','Body','Mileage','EngineV','Engine Type']]


# In[214]:


x.head()


# In[215]:


x.describe()


# In[216]:


from sklearn.preprocessing import LabelEncoder


# In[217]:


data


# In[218]:


le=LabelEncoder()
label1=le.fit_transform(data['Brand'])
label2=le.fit_transform(data['Body'])
label3=le.fit_transform(data['Engine Type'])
label4=le.fit_transform(data['Model'])


# In[219]:


label4


# In[220]:


Newdata=data.drop(["Brand","Body","Engine Type","Model"],axis="columns")


# In[221]:


Newdata.head()


# In[222]:


Newdata["Brand"]=label1


# In[223]:


Newdata["Body"]=label2


# In[224]:


Newdata["Engine Type"]=label3


# In[225]:


Newdata["Model"]=label4


# In[226]:


Newdata


# In[227]:


Y=Newdata["Price"]


# In[228]:


X=Newdata[['Brand','Body','Mileage','EngineV','Engine Type','Model']]
X.head()


# In[229]:


from sklearn.tree import DecisionTreeRegressor


# In[230]:


model=DecisionTreeRegressor(random_state=1)


# In[231]:


model.fit(X,Y)


# In[232]:


print("Making predictions for the following 5 cars:")
print(X.head())
print("The predictions are")
print(model.predict(X.head()))


# In[233]:


Y.head()


# In[234]:


pd.set_option('display.max_rows', 25)


# In[235]:


data.loc[Newdata['Price'] == max(Newdata['Price'])]


# In[236]:


print(max(Newdata['Price']))


# In[237]:


Newdata.loc[Newdata['Price'] == max(Newdata['Price'])]


# In[238]:


testdata=pd.read_csv("carTest.csv")
testdata.columns=['Brand','Body','Mileage','EngineV','Engine Type','Model']
testdata


# In[239]:


print("Making predictions for the following 8 cars:")
print(testdata)
print("The predictions are")
print(model.predict(testdata))


# In[240]:


import matplotlib
import matplotlib.pyplot as plt


# In[241]:


Newdata["Brand"]


# In[242]:


pd.options.display.max_rows


# In[243]:


pd.options.display.max_rows = 4329


# In[244]:


Newdata["Brand"].head()


# In[245]:


Newdata.plot(x="Price",y=["Brand","Body","Mileage","Model"])


# In[246]:


Newdata.plot(y="Price",color='r')


# In[247]:


Newdata.plot("Price")


# In[248]:


data[1:40].plot(y="Price",x="Brand",kind='bar')

