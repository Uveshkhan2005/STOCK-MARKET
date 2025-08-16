#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


ompany_name = 'AAPL'  
data = pd.read_csv(f'.build/{'AAPL'}_last_200_days_cleandata.csv')


# In[4]:


# LINE PLOT OF STOCK PRICES OVER TIME :
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Close'], label='Close Price')  
plt.title(f'{'AAPL'} Stock Price Over Last 200 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('VISUALIZATION IMG1.png') 
plt.show()


# In[5]:


# PLOT VOLUME TREND OVER TIME :

plt.figure(figsize=(12,4))
plt.bar(data['Date'], data['Volume'], color='orange')
plt.title(f'{'AAPL'} Trading Volume Over Last 200 Days')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.savefig('VISUALIZATION IMG2.png') 
plt.show()


# In[6]:


# PLOT MOVING AVERANGES (7 AND 14 DAYS ) :

plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.plot(data['Date'], data['ma_7'], label='7-day MA')
plt.plot(data['Date'], data['ma_14'], label='14-day MA')
plt.title(f'{'AAPL'} Close Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('VISUALIZATION IMG3.png') 
plt.show()


# In[7]:


# HISTOGRAM OF DAILY RETURNS :

plt.figure(figsize=(8,5))
sns.histplot(data['return_pct'].dropna(), bins=50, kde=True)
plt.title(f'{'AAPL'} Daily Return Percentage Distribution')
plt.xlabel('Daily Return %')
plt.ylabel('Frequency')
plt.savefig('VISUALIZATION IMG4.png') 
plt.show()


# In[8]:


# CORRELATION HEATMAP :

numeric_data = data.select_dtypes(include=['number'])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title(f'{'AAPL'} Feature Correlation Matrix')
plt.savefig('VISUALIZATION IMG5.png') 
plt.show()


# In[9]:


data.to_csv('Data Visualization main.csv', index=False)






