import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree    # Used to plot tree visual, xgboost has plot_tree method too


clusters = pd.read_parquet('cluster_lables_20.pq')

DT = DecisionTreeClassifier(max_depth=3)    # set about 2-5 range
clf = OneVsRestClassifier(DT, n_jobs=-1)
# RT = RandomForestClassifier(n_estimators=30)
# clf = OneVsRestClassifier(RT, n_jobs=-1)

features = list(clusters.drop(columns=['acct_oid','Unnamed: 0','hdbsca_probability','hdbscan_clust','kmeans_clust']).columns)
X = clusters[features].to_numpy()
Y = clusters['kmeans_clust'].to_numpy()

clf.fit(X,Y)

print(classification_report(Y, clf.predict(X)))


Path('kmeans_plots').mkdir(exist_ok=True)

est1 = clf.estimators_[1]
temp_df = pd.DataFrame({'singe_tree': est1.predict(X), 'ensemble': clf.predict(X)})
temp_df[temp_df['ensemble'] == 1].head()

for i, est in enumerate(clf.estimators_):
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

    plt.figure(figsize=(16,16))
    plot_tree(est, feature_names=features, class_names=['Not in CLuster', 'In Cluster'], filled=True)
    plt.title(f'Tree {i}, Cluster {i}')
    plt.savefig(f'kmeans_plots/tree{i}.png')
    plt.show()


kmeans_features = []
kmeans_dict = {}

features = np.array(feature_cols)
for i, est in enumerate(clf.estimators_):
    importances = est.feature_importances_
    mask = np.where(importances >= 0.1)
    clust_features = features[mask]
    
    kmeans_features = set(kmeans_features) | set(clust_features)
    
    kmeans_dict[f'cluster_{i}'] = list(clust_features)
 
kmeans_dict['important_features'] = list(kmeans_features)


with open('kmeans.json', 'w') as fp:
    json.dump(kmeans_dict, fp)
