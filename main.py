#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


houseData = {
    
    "area (sq.f)" :[100,150,300,400,550,],
    "price (LKR lakhs)" : [100,140,250,340,400]
    
    
}

df = pd.DataFrame(houseData)
print(df)


# In[9]:



import matplotlib.pyplot as plt
plt.scatter(df["area (sq.f)"],df["price (LKR lakhs)"])
plt.xlabel("area (sq.f)")
plt.ylabel("price (LKR lakhs")


# In[10]:


from sklearn.linear_model import LinearRegression


# In[23]:


X=df[["area (sq.f)"]]


# In[24]:


y=df[["price (LKR lakhs)"]]


# In[25]:


P_model= LinearRegression()
P_model.fit(X,y)


# In[26]:


print(f"Coefficient : {P_model.coef_[0]} ")


# In[27]:


print(f"intercept : {P_model.intercept_} ")


# In[30]:


#prediction 


def predict_price(area_):
    predicted_price = P_model.predict(area_)
    return predicted_price


# In[32]:


#example

area_1 = [[180]]
predicted_prices = predict_price(area_1)


# In[36]:


print(f"predicted house price is  {predicted_prices[0]} ")


# In[45]:


plt.scatter(X,y)
plt.plot(X,P_model.predict(X), color ="green", label ="Regression Line")
plt.xlabel("area (sq.f)")
plt.ylabel("price (LKR lakhs")
plt.legend()


# In[ ]:




