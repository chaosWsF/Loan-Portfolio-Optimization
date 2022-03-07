#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


# In[2]:


df = pd.read_csv('kiting_monthly_feat_clust.csv')


# In[3]:


df.drop(columns = ['deposits_db','payment_loan_db','payment_loan_cr','bank_transfer_debit_cr',
                  'e_transfer_cr','installment_payment_cr','withdrawal_ABM_cr','withdrawal_interac_abm_cr',
                   'mastercard_bill_cr','withdrawal_cr'], inplace = True)


# In[4]:


df.head()



# In[8]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree # Used to plot tree visual, xgboost has plot_tree method too
from sklearn.tree import export_text # Use to extract SQL/ logic


# In[47]:


DT = DecisionTreeClassifier(max_depth = 3) # set about 2-5 range

clf = OneVsRestClassifier(DT, n_jobs=-1)


# In[58]:


X = df.drop(columns = ['acct_oid','Unnamed: 0','hdbsca_probability','hdbscan_clust','kmeans_clust']).values

Y = df['kmeans_clust'].values


# In[59]:


clf.fit(X,Y)


# In[50]:


print(classification_report(Y,clf.predict(X)))


# In[51]:


import os
os.makedirs('kmeans_plots',exist_ok = True)


# In[52]:


#Verifying that the n_th tree coressponds to the n_th cluster
est1 = clf.estimators_[1]
temp_df = pd.DataFrame({'singe_tree':est1.predict(X),'ensemble':clf.predict(X)})
temp_df[temp_df['ensemble'] == 1].head()


# In[53]:


features = list(df.drop(columns = ['acct_oid','Unnamed: 0','hdbsca_probability','hdbscan_clust','kmeans_clust']).columns)
for i,est in enumerate(clf.estimators_):
    importances = est.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure()
    plt.title(f'Feature Importances, Tree {i}, Cluster {i}')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(f'kmeans_plots/cluster{i}.png')
    plt.show()
    plt.close()

    plt.figure(figsize = (16,16))
    plot_tree(est,feature_names = features, class_names = ['Not in CLuster', 'In Cluster'], filled = True)
    plt.title(f'Tree {i}, Cluster {i}')
    plt.savefig(f'kmeans_plots/tree{i}.png')
    plt.show()


# In[67]:


import json
kmeans_features = []

kmeans_dict = {}

features = np.array(features)
for i,est in enumerate(clf.estimators_):
    importances = est.feature_importances_
    mask = np.where(importances >=0.1)
    clust_features = features[mask]
    
    kmeans_features = set(kmeans_features) | set(clust_features)
    
    kmeans_dict[f'cluster_{i}'] = list(clust_features)
    
    
    
kmeans_dict['important_features'] = list(kmeans_features)
    


# In[68]:




with open('kmeans.json', 'w') as fp:
    json.dump(kmeans_dict, fp)




