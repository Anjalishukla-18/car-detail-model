#!/usr/bin/env python
# coding: utf-8

# In[4]:


#q2


# In[7]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[8]:


df=pd.read_csv('C:\\Users\\Anmol\\Downloads\\CAR DETAILS.csv')
print(df)


# In[9]:


df.describe()


# In[10]:


df['owner'].value_counts()


# In[11]:


df['name']


# In[12]:


df['fuel'].value_counts()


# In[13]:


df.shape


# In[14]:


df.dtypes


# In[15]:


df.drop_duplicates(inplace=True)


# In[16]:


df.duplicated().sum()


# In[17]:


df.info()


# In[18]:


#q3


# In[19]:


df.isnull().sum()


# In[20]:


from sklearn.preprocessing import OneHotEncoder
ohc=OneHotEncoder()


# In[21]:


one_hot_encoder=ohc.fit_transform(df[['fuel','seller_type','transmission','owner']]).toarray()
one_hot_encoder


# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


one_hot_encoder.shape


# In[24]:


sc=StandardScaler()
c=sc.fit_transform(df[['year','km_driven']])
print(c.shape)
print(type(c))


# In[25]:


encoded_df = pd.DataFrame(one_hot_encoder)
scaled_df = pd.DataFrame(c)


# In[26]:


scaled_df


# In[27]:


scaled_df.columns = ['year', 'km_driven']


# In[28]:


combined_df = pd.concat([encoded_df, scaled_df], axis=1)


# In[29]:


combined_df


# In[ ]:





# In[30]:


#q4


# In[31]:


corr_matrix = df[['year','selling_price','km_driven']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[32]:


numerical_features = ['year', 'selling_price']

for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Yearly Selling Price Distribution for {feature}')
    plt.xlabel('Year')
    plt.ylabel('Selling Price')
    plt.show()


# In[33]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='year', y='selling_price', data=df)
plt.title(f'Boxplot of sellind price yearly')
plt.xlabel('year')
plt.ylabel('selling_price')
plt.show()


# In[34]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='year', y='selling_price', data=df)
plt.title('Scatter Plot of selling price yearly')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()


# In[35]:


sns.pairplot(df)
plt.title('Pairplot of Numerical Features')
plt.show()


# In[36]:


#q5


# In[37]:


x=combined_df
y=df['selling_price']
print(type(x))
print(type(y))
print(x.shape)
print(y.shape)


# In[38]:


from sklearn.model_selection import train_test_split 


# In[39]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[40]:


x_train.columns = x_train.columns.astype(str)


# In[41]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso
lr=LinearRegression()
model=lr.fit(x_train,y_train)


# In[42]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred_test)


# In[51]:


mse


# In[52]:


x_train


# In[53]:


x_test.columns = x_train.columns.astype(str)


# In[54]:


y_pred=model.predict(x_test)
y_pred_train=model.predict(x_train)
y_pred_test=model.predict(x_test)


# In[55]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[56]:


def reg_eval_metrics(x_pred,x_pred_test):
    mse=mean_square_erroe(ytest,ypred)
    mae=mean_absolute_error(xtrain,xpred)
    rmse=np.sqrt(mean_squared_error(ytest,ypred))
    
    print("mse:",mse)
    print("mae:",mae)
    print("rmse:",rmse)


# In[57]:


RD=Ridge(alpha=0.4)
RD.fit(x_train,y_train)


# In[58]:


from sklearn.metrics import mean_squared_error
mse_ridge = mean_squared_error(y_test,y_pred_test)


# In[59]:


mse_ridge


# In[60]:


LS=Lasso(alpha=15.7)
LS.fit(x_train,y_train)


# In[61]:


from sklearn.metrics import mean_squared_error
mse_lasso = mean_squared_error(y_test,y_pred_test)


# In[62]:


mse_lasso


# In[63]:


from sklearn.neighbors import KNeighborsRegressor


# In[64]:


KNN=KNeighborsRegressor(n_neighbors=15)
KNN.fit(x_train,y_train)


# In[65]:


from sklearn.metrics import mean_squared_error
mse_KNN= mean_squared_error(y_test,y_pred_test)


# In[66]:


mse_KNN


# In[67]:


min(mse,mse_ridge,mse_lasso,mse_KNN)


# In[68]:


from sklearn.metrics import accuracy_score


# In[69]:


#q7


# In[70]:


df=pd.read_csv('C:\\Users\\Anmol\\Downloads\\CAR DETAILS.csv')
print(df)


# In[75]:


get_ipython().system('pip install pickle')


# In[76]:


import pickle


# In[77]:


with open('KNN.pkl', 'wb') as file:
    pickle.dump(KNN, file)


# In[78]:


with open('Ridge.pkl', 'wb') as file:
    pickle.dump(RD, file)


# In[79]:


with open('Lasso.pkl', 'wb') as file:
    pickle.dump(LS,file)


# In[80]:


with open('car_details.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# In[81]:


y_pred = loaded_model.predict(x_test)


# In[82]:


#q8


# In[83]:


y_pred


# In[84]:


selected_data = x.sample(n=20)


# In[85]:


y_pred = loaded_model.predict(selected_data)


# In[ ]:


x.columns = x.columns.astype(str)


# In[ ]:


x_test


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




