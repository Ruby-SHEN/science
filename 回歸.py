#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import plot_tree

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#####
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder


# In[2]:


# 載入檔案
df = pd.read_csv('CarPrice_Assignment.csv')
df.head()


# In[3]:


df.info()


# In[4]:


# 資料清理 清掉類別型態資料 (只能有連續變項)
df = df.drop(['car_ID'], axis=1)
df = df.drop(['CarName'], axis=1) #汽車公司名稱
df = df.drop(['fueltype'], axis=1) #汽車燃料類型，即汽油或柴油
df = df.drop(['aspiration'], axis=1) #汽車用吸塵器
df = df.drop(['doornumber'], axis=1) #汽車的門數
df = df.drop(['carbody'], axis=1) #車體
df = df.drop(['drivewheel'], axis=1) #主動輪類型
df = df.drop(['enginelocation'], axis=1) #汽車發動機的位置
df = df.drop(['enginetype'], axis=1) #發動機類型
df = df.drop(['cylindernumber'], axis=1) #放置在汽車中的氣瓶
df = df.drop(['fuelsystem'], axis=1) #汽車燃油系統
df = df.drop(['boreratio'], axis=1) #汽車鑽孔


# 使用LabelEncoder將標籤轉為對應數值
lb = LabelEncoder()
#lb.fit(df['CarName'].drop_duplicates()) 
#df['CarName'] = lb.transform(df['aspiration'])

#lb.fit(df['fueltype'].drop_duplicates()) 
#df['fueltype'] = lb.transform(df['aspiration'])

#lb.fit(df['aspiration'].drop_duplicates()) 
#df['aspiration'] = lb.transform(df['aspiration'])

#lb.fit(df['doornumber'].drop_duplicates()) 
#df['doornumber'] = lb.transform(df['doornumber'])

#lb.fit(df['carbody'].drop_duplicates()) 
#df['carbody'] = lb.transform(df['carbody'])

#lb.fit(df['drivewheel'].drop_duplicates()) 
#df['drivewheel'] = lb.transform(df['drivewheel'])

#lb.fit(df['enginelocation'].drop_duplicates()) 
#df['enginelocation'] = lb.transform(df['enginelocation'])

#lb.fit(df['enginetype'].drop_duplicates()) 
#df['enginetype'] = lb.transform(df['enginetype'])

#lb.fit(df['cylindernumber'].drop_duplicates()) 
#df['cylindernumber'] = lb.transform(df['cylindernumber'])

#lb.fit(df['fuelsystem'].drop_duplicates()) 
#df['fuelsystem'] = lb.transform(df['fuelsystem'])

df.head()


# In[5]:


# 相關係數
corr = df.corr()
large = 15 ;med = 15; small = 15
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (15, 15),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
# corr.style.background_gradient(cmap='coolwarm').set_precision(2)
sns.heatmap(corr, annot=True)


# In[6]:


from sklearn.model_selection import train_test_split

# 選擇需要欄位
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 切分資料為 訓練8:測試2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# 載入模型
import math
ln = LinearRegression()
ln.fit(X_train, y_train)

# 預測
ln_predict = ln.predict(X_test)

num_data = X_train.shape[0]

# 模型效能指標
ln_mse = mean_squared_error(y_test, ln_predict, squared=False)
ln_rse = math.sqrt(ln_mse/(num_data-2))
ln_r2 = r2_score(y_test, ln_predict)

print("ln_mse={}".format(ln_mse))
print("ln_rse={}".format(ln_rse))
print("ln_r2={}".format(ln_r2))


# In[8]:


import math
rfr = RandomForestRegressor(random_state=42)
rfr.fit(X_train, y_train)

num_data = X_train.shape[0]

rfr_predict = rfr.predict(X_test)
rfr_mse = mean_squared_error(y_test, rfr_predict, squared=False)
rfr_rse = math.sqrt(rfr_mse/(num_data-2))
rfr_r2 = r2_score(y_test, rfr_predict)

print("rfr_mse={}".format(rfr_mse))
print("rfr_rse={}".format(rfr_rse))
print("rfr_r2={}".format(rfr_r2))


# In[9]:


X_train.columns


# In[10]:


f, ax = plt.subplots(figsize=(20, 10))
plot_tree(rfr.estimators_[0], max_depth=2, ax=ax, feature_names=X_train.columns)


# In[11]:


import math
ann = MLPRegressor(batch_size=10 ,learning_rate_init=1e-1 ,random_state=42)
ann.fit(X_train, y_train)

num_data = X_train.shape[0]

ann_predict = ann.predict(X_test)
ann_mse = mean_squared_error(y_test, ann_predict, squared=False)
ann_rse = math.sqrt(ann_mse/(num_data-2))
ann_r2 = r2_score(y_test, ann_predict)

print("ann_mse={}".format(ann_mse))
print("ann_rse={}".format(ann_rse))
print("ann_r2={}".format(ann_r2))


# In[12]:


get_ipython().system('pip install eli5')
import eli5 as eli5
eli5.show_weights(rfr, feature_names=X_train.columns.tolist())


# In[13]:


from eli5 import show_prediction 
eli5.show_prediction(rfr, X_train.iloc[1],show_feature_values=True)


# In[14]:


import eli5 as eli5
eli5.show_weights(ln, feature_names=X_train.columns.tolist())


# In[15]:


from eli5 import show_prediction 
eli5.show_prediction(ln, X_train.iloc[1],show_feature_values=True)


# In[ ]:




