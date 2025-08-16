#!/usr/bin/env python
# coding: utf-8

# In[1]:


# DATA PREPROCESSING :

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# LOAD DATASET :
data = pd.read_csv('MAIN/STOCK DETAILS.csv')


# In[3]:


# Quick look at the data
print(data.head())
print(data.info())


# In[4]:


company_name = 'AAPL'  

# Filter rows for this company only
data_company = data[data['Company'] == 'AAPL'].copy()

print(f"Total rows for {'AAPL'}: {len(data_company)}")


# In[5]:


# converts date column to date and time formate :
data_company['Date'] = pd.to_datetime(data_company['Date'], utc=True)
data_company['Date'] = data_company['Date'].dt.tz_convert(None)


# In[6]:


# sort data by 'date' and reset index:
data_company.sort_values('Date', inplace=True)
data_company.reset_index(drop=True, inplace=True)


# In[7]:


# checking last 200 rows iof data and its shape :
data_200 = data_company.tail(200).copy()
print(f"Data shape for last 200 days: {data_200.shape}")


# In[8]:


# check for missing values :
print(data_200.isnull().sum())


# In[9]:


# calculating daitly price change nd average og 'close' price :
data_200['price_change'] = data_200['Close'] - data_200['Open']  
data_200['return_pct'] = data_200['Close'].pct_change()


# In[10]:


# 7 days nd 14 days moving averge :
data_200['ma_7'] = data_200['Close'].rolling(window=7).mean()
data_200['ma_14'] = data_200['Close'].rolling(window=14).mean()


# In[13]:


data_200.to_csv(f'{'AAPL'}_last_200_days_cleandata.csv', index=False)


# In[15]:


data_company.to_csv('Data PreProcessing main.csv', index=False)







