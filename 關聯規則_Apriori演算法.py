#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 下載套件
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


# In[2]:


# 載入檔案
store_data = pd.read_csv('Groceries_dataset.csv')
store_data.head()


# In[3]:


store_data.shape


# In[4]:


# 將格式轉為list
transactions = []
for i in range(0, len(store_data)):
    transactions.append([str(j) for j in store_data.iloc[i,:] if str(j)!='nan'])

len(transactions)


# In[5]:


association_rules = apriori(transactions, min_support=0.000326, min_confidence=0.2, min_length=2)
association_results = list(association_rules)
print(association_results)


# In[6]:


print(association_results[0])


# In[7]:


for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

