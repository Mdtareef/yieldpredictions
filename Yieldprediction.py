#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\Md Tareef\Desktop\tarif data\tarif project\yield_df.csv\yield_df.csv")
df


# In[3]:


df.drop('Unnamed: 0', axis=1, inplace=True)


# In[4]:


df.shape


# In[5]:


df.isna().sum().sum()


# In[6]:


df.duplicated().sum()


# In[7]:


df.describe()


# In[8]:


df.drop_duplicates(inplace=True)


# In[9]:


df.duplicated().sum()


# In[10]:


df.corr()


# In[11]:


df['average_rain_fall_mm_per_year']


# In[12]:


def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True


# In[13]:


to_drop=df[df['average_rain_fall_mm_per_year'].apply(isStr)].index


# In[14]:


df=df.drop(to_drop)


# In[15]:


df


# In[16]:


plt.figure(figsize=(10,25))
sns.countplot(y=df['Area'])


# # yield_per_country

# In[17]:


df.info()


# In[18]:


country=(df['Area'].unique())


# In[19]:


for i in country:
    print(i)


# In[20]:


yield_per_country=[]
for i in country:
    yield_per_country.append(df[df['Area']==i]['hg/ha_yield'].sum())


# In[21]:


(df['hg/ha_yield'].unique())


# In[22]:


plt.figure(figsize=(10,20))
sns.barplot(y=country, x=yield_per_country)


# In[23]:


df['Item'].value_counts()


# In[24]:


sns.countplot(y=df['Item'])


# In[25]:


crops=(df['Item'].unique())
crops


# In[26]:


yield_per_crop=[]
for crop in crops:
    yield_per_crop.append(df[df['Item']==crop]['hg/ha_yield'].sum())


# In[27]:


yield_per_crop


# In[28]:


sns.barplot(y=crops,x=yield_per_crop)


# In[29]:


df.columns


# In[30]:


col=['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp','Area', 'Item','hg/ha_yield']
df=df[col]


# In[31]:


df


# In[32]:


X=df.drop('hg/ha_yield',axis=1)
y=df['hg/ha_yield']


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


X_train.shape


# In[36]:


X_test.shape


# In[37]:


X_train


# In[38]:


from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[39]:


ohe=OneHotEncoder(drop='first')
scaler=StandardScaler()


# In[40]:


preprocessor=ColumnTransformer(
 transformers=[('onehotencoder',ohe,[4,5]), ('standerization',scaler,[0,1,2,3])],remainder='passthrough'
)


# In[41]:


preprocessor


# In[42]:


X_train_dummy=preprocessor.fit_transform(X_train)
X_test_dummy= preprocessor.transform(X_test)


# In[43]:


X_train_dummy


# In[44]:


from sklearn.utils.fixes import _sparse_linalg_cg


# In[45]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor  # Corrected import statement
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[46]:


models={
    'lr':LinearRegression(),
    'lss':Lasso(),
    'rg':Ridge(),
    'Knr':KNeighborsRegressor(),
    'dtr':DecisionTreeRegressor()
}

for name, mod in models.items():
    mod.fit(X_train_dummy,y_train)
    y_pred=mod.predict(X_test_dummy)
    
    print(f"{name} MSE : {mean_squared_error(y_test,y_pred)} Score {r2_score(y_test,y_pred)}")


# In[47]:


dtr=DecisionTreeRegressor()
dtr.fit(X_train_dummy,y_train)
dtr.predict(X_test_dummy)


# In[ ]:





# In[56]:


def prediction(Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item):    
    features=np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]])
    
    transformed_features=preprocessor.transform(features)
    predicted_value=dtr.predict(transformed_features).reshape(1,-1)
    return predicted_value[0]


# In[ ]:





# In[57]:


Year= 2000
average_rain_fall_mm_per_year=59.0
pesticides_tonnes=3024.11
avg_temp=26.55
Area='Saudi Arabia'
Item='Sorghum'

prediction(Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item)


# In[69]:


import pickle
pickle.dump(dtr,open('dtr.pkl','wb'))
#pickle.dump(preprocessor,('preprocessor.pkl','wb'))


# In[70]:


#with open('preprocessor.pkl', 'wb') as file:
   # pickle.dump(preprocessor, file)
pickle.dump(preprocessor,open('preprocessor.pkl','wb'))


# In[ ]:




