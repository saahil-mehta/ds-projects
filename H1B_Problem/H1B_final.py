#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 


# In[2]:


df = pd.read_csv("H-1B_Disclosure_Data_FY17.csv")


# In[3]:


df


# In[4]:


df.CASE_STATUS.value_counts()


# In[5]:


(df.isnull().sum()/len(df))*100


# In[ ]:


# df.drop(columns=["Unnamed: 0,EMPLOYER_BUSINESS_DBA,ORIGINAL_CERT_DATE,EMPLOYER_PROVINCE,EMPLOYER_PHONE_EXT,"])


# In[6]:


df1 = df.dropna(thresh=df.shape[0]*0.6,how='all',axis=1)
df1=df1.drop(columns=["Unnamed: 0"])


# In[ ]:


df1


# In[7]:


df.shape,df1.shape


# In[8]:


(df1.isnull().sum()/len(df1))*100


# In[ ]:


#_____________-removed all features with values of null above 60%


# In[9]:


df1.dtypes


# In[10]:


df1["DECISION_TIME"]=(pd.to_datetime(df1['DECISION_DATE'])-pd.to_datetime(df1['CASE_SUBMITTED'])).dt.days


# In[11]:


df1["EMP_TIME"]=(pd.to_datetime(df1['EMPLOYMENT_END_DATE'])-pd.to_datetime(df1['EMPLOYMENT_START_DATE'])).dt.days


# In[ ]:


### next drop dec date , case submitted , emp start date , emp end date


# In[12]:


df1.columns


# In[13]:


numerical = df1.select_dtypes(include = np.number)
nonnumerical = df1.select_dtypes(exclude = np.number)


# In[14]:


cols = list(numerical)
cols


# In[15]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[16]:


# numerical["CASE_STATUS"]= nonnumerical.CASE_STATUS
# numerical["VISA_CLASS"] = nonnumerical.VISA_CLASS


# In[18]:


numerical = scaler.fit_transform(numerical)
numerical = pd.DataFrame(numerical)


# In[19]:


numerical.columns = cols


# In[20]:


cols


# In[21]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[22]:


nonnumerical


# In[23]:


list(nonnumerical.iloc[:,:])


# In[24]:


nonnumerical1 = nonnumerical.copy(deep=True)


# In[25]:


nonnumerical1


# In[26]:


nonnumerical=nonnumerical.drop(columns=["CASE_NUMBER","CASE_SUBMITTED","DECISION_DATE","EMPLOYMENT_START_DATE"                          ,"EMPLOYMENT_END_DATE","EMPLOYER_NAME",'EMPLOYER_ADDRESS','EMPLOYER_CITY',                           'EMPLOYER_POSTAL_CODE','EMPLOYER_COUNTRY',                          'AGENT_REPRESENTING_EMPLOYER','SOC_CODE',                          'WORKSITE_CITY','WORKSITE_COUNTY','WORKSITE_POSTAL_CODE'])


# In[ ]:


# nonnumerical.SOC_NAME = nonnumerical1.SOC_NAME


# In[ ]:


# nonnumerical.drop(columns=["SOC_CODE"])


# In[ ]:


nonnumerical=nonnumerical.drop(columns=["EMPLOYER_PHONE",""


# In[27]:


list(numerical.iloc[:,:-1])


# In[28]:


list(nonnumerical.iloc[:,:-1])


# In[29]:


nonnumerical.dtypes


# In[31]:


nonnumerical1.SOC_NAME.value_counts().nunique()


# In[32]:


## soc_code has punctuations so soc_name has been taken


# In[36]:


nonnumerical.columns


# In[37]:


## LabelEncode After FS
for x in list(nonnumerical.iloc[:,:]):
    nonnumerical[x] = le.fit_transform(nonnumerical[x].astype(str))


# In[39]:


numerical


# In[40]:


df_con = pd.concat([numerical,nonnumerical],1)


# In[41]:


df_con.isnull().sum()


# In[42]:


df_con = df_con.replace(np.nan, 0)


# In[46]:


X = df_con.drop(columns=["CASE_STATUS"])
y = df_con["CASE_STATUS"]


# In[44]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()


# In[47]:


from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.3, random_state = 42)


# In[48]:


dtc.fit(X_train,y_train)


# In[49]:


y_pred = dtc.predict(X_test)


# In[50]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)


# In[52]:


coef3 = pd.Series(dtc.feature_importances_,X.columns).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')


# In[59]:


feat_importances = pd.Series(dtc.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')


# In[55]:


from sklearn.model_selection import cross_val_score


# In[56]:


cross_val_score(dtc,X,y)


# In[83]:


from sklearn.model_selection import GridSearchCV
params= {"max_depth":np.arange(1,10),"min_samples_leaf":np.arange(0.1,0.5,5),"min_samples_split":np.arange(0.1,1.0)}

dgrid_cv= GridSearchCV(dtc,params,cv=5)


# In[84]:


dgrid_cv


# In[85]:


dgrid_cv.fit(X_train,y_train)
dgrid_cv.best_score_


# In[72]:


dtc.get_params().keys()


# In[87]:


dgrid_cv.best_params_, dgrid_cv.best_score_


# In[89]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:




