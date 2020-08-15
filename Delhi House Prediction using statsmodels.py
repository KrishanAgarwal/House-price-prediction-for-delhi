#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

print("Import successful")


# In[2]:


housing=pd.read_csv("housing.csv")
housing.head()


# In[3]:


housing.shape


# In[4]:


housing.info()


# In[5]:


housing.describe()


# In[6]:


sns.pairplot(data=housing)
plt.show()


# In[7]:


plt.figure(figsize=(20,12))
plt.subplot(2,3,1)
sns.boxplot(x="mainroad",y="price",data=housing)
plt.subplot(2,3,2)
sns.boxplot(x="guestroom",y="price",data=housing)
plt.subplot(2,3,3)
sns.boxplot(x="basement",y="price",data=housing)
plt.subplot(2,3,4)
sns.boxplot(x="hotwaterheating",y="price",data=housing)
plt.subplot(2,3,5)
sns.boxplot(x="airconditioning",y="price",data=housing)
plt.subplot(2,3,6)
sns.boxplot(x="furnishingstatus",y="price",data=housing)
plt.show()


# In[8]:


housing.groupby("mainroad")["mainroad"].count()


# In[9]:


housing["mainroad"].value_counts()


# In[9]:


housing.columns


# In[10]:


varlist=["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
housing[varlist]=housing[varlist].apply(lambda x: x.map({"yes":1, "no":0}))
housing[varlist].head()


# In[11]:


housing.head()


# In[12]:


status=pd.get_dummies(housing["furnishingstatus"],drop_first=True)
status.head()


# <h5> we droped one column from status because we can interpret easily using two columns and therefore we reduced the redundancy</h5>

# 00 - furnished;
# 10 - Semifurnished;
# 01 - Unfurnished

# Now add the status column into dataframe. We can do it using concat function

# In[13]:


housing=pd.concat([housing,status],axis=1)
housing.head()


# In[14]:


housing.drop("furnishingstatus",axis=1,inplace=True)
housing.head()


# In[15]:


df_train, df_test=train_test_split(housing,test_size=0.3,random_state=100)
print(df_train.shape)
print(df_test.shape)


# <h5> Rescaling the features </h5>

# <p> Here we can see that the area variable has very large value compared to other variables in the dataset.<br> So we need to rescale the data based on two factors :- <br>
# 1) Min-max scaling<br>
# 2) Standardisation(mean-0, sigma-1) 
# 
# Min-max scaling - Normalization (x-xmin)/(xmax-xmin)<br>
# Standardisation - (x-mu)/sigma
# 
# </p> 
# 

# In[16]:


#instantiate an object

scaler=MinMaxScaler()

# we will use minmaxscaler on numeric datapoints
# create a list of numeric variables

num_vars=["area","bathrooms","bedrooms","stories","parking","price"]

df_train[num_vars]=scaler.fit_transform(df_train[num_vars])
df_train.head()


# In[17]:


df_train[num_vars].describe()


# <h5> Training the model</h5>

# In[18]:


plt.figure(figsize=(20,12))
sns.heatmap(df_train.corr(),annot=True,cmap="RdGy")
plt.show()


# In[19]:


df_train.head()


# In[20]:


y_train=df_train.pop('price')
x_train=df_train


# In[21]:


x_train.head()


# In[25]:


# model building 
# The approach is to add one-one variables and make the model.
#First thing in statsmodels is that we need to add constant 

x_train_sm=sm.add_constant(x_train['area'])

#create first mode
lr=sm.OLS(y_train,x_train_sm)

#fit the model

lr_model=lr.fit()

# parameters
lr_model.params


# In[26]:


lr_model.summary()


# In[27]:


# add another variable and keep on doing until we add all the variables
# we can select any variable but better go with correlation.

x_train_sm=x_train[['area','bathrooms']]
x_train_sm=sm.add_constant(x_train_sm)

#create model
lr=sm.OLS(y_train,x_train_sm)

#fit the mode

lr_model=lr.fit()

#summary

lr_model.summary()


# In[28]:


x_train_sm=x_train[["area","bathrooms","bedrooms"]]
x_train_sm=sm.add_constant(x_train_sm)

#create model
lr=sm.OLS(y_train,x_train_sm)

#fit the mode

lr_model=lr.fit()

#summary

lr_model.summary()


# Add one one variable is a tedious approach so we will use another method
# <p> we will add all the variables and remove the variables one by one that are not needed based on some values like p-value,<br> Adjusted r-square, r-square, VIF etc.</p>
# 

# In[22]:


#Build a model using all the variables 
x_train_sm=sm.add_constant(x_train)

lr=sm.OLS(y_train,x_train_sm)

lr_model=lr.fit()


lr_model.summary()


# <p>We can remove the varibales based on two parameters</p>
# <p>significance</p>
# <p>VIF <br> VIF > 5 is very high because if we substitute the VIF value in the VIF equation we will get r2 as 0.8 which <br> is very high and it leads to multicollinearity so we drop such variables </p>
# 

# In[23]:


# create a dataframe that contains the names of all the features and their respective VIF values.

vif=pd.DataFrame()
vif["features"]=x_train.columns
vif["VIF"]=[variance_inflation_factor(x_train.values,i) for i in range(x_train.shape[1])]
vif["VIF"]=round(vif["VIF"],2)
vif=vif.sort_values(by="VIF", ascending=False)
vif


# <p> Always drop one variable at a time because it might happen that dropping one variable may reduce the Vif of other variables</p>

# <p>Now to drop one variable we see VIF and p-value
# we will drop the variable which has high p-value rather than VIF. Dropping the variable will change the VIF value.</p>

# In[24]:


x=x_train.drop("semi-furnished",axis=1)


# In[25]:


x_train_sm=sm.add_constant(x)

lr=sm.OLS(y_train,x_train_sm)

lr_model=lr.fit()

lr_model.summary()


# In[26]:


vif=pd.DataFrame()
vif["features"]=x.columns
vif["VIF"]=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
vif["VIF"]=round(vif["VIF"],2)
vif=vif.sort_values(by="VIF", ascending=False)
vif


# In[27]:


x=x.drop("bedrooms",axis=1)


# In[28]:


x_train_sm=sm.add_constant(x)

lr=sm.OLS(y_train,x_train_sm)

lr_model=lr.fit()

lr_model.summary()


# In[29]:


vif=pd.DataFrame()
vif["features"]=x.columns
vif["VIF"]=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
vif["VIF"]=round(vif["VIF"],2)
vif=vif.sort_values(by="VIF", ascending=False)
vif


# In[31]:


y_train_pred=lr_model.predict(x_train_sm)


# In[32]:


res=y_train-y_train_pred
sns.distplot(res)


# In[33]:


sns.distplot(y_train, hist=False, color="r",label="Actual Value")
sns.distplot(y_train_pred, hist=False, color="b",label="Predicted value")


# <h2> Prediction and evaluation on testset</h2>

# <p>fit is not applied on test dataset because the test dataset might change.<br> The test dataset is new everytime so we don't fit the test dataset. Fitting is done on train dataset</p>
# 

# In[34]:


num_vars=["area","bathrooms","bedrooms","stories","parking","price"]

df_test[num_vars]=scaler.transform(df_test[num_vars])
df_test.head()


# In[35]:


df_test.describe()


# In[36]:


y_test=df_test.pop("price")
x_test=df_test


# In[37]:


x_test.head()


# In[38]:


x_test_sm=sm.add_constant(x_test)


# In[ ]:


#since we have droppped three variables in the train dataset so it will throw an error saying that there are not 
#coefficient for the following three variables/columns.
# drop the three variables


# In[39]:


x_test_sm=x_test_sm.drop(["bedrooms","semi-furnished"],axis=1)


# In[40]:


# predict 
y_test_pred=lr_model.predict(x_test_sm)


# <h5> Evaluate the model</h5>

# In[41]:


r2_score(y_test,y_test_pred)


# In[ ]:





# In[ ]:





# In[ ]:




