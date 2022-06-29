# Loan-Portfolio-Optimization

## Clustering Pipeline
    
    1. Provide the settings of the experiment in `config.py`
    1. Run `generating_raw_data.py` first to get cleaned `loans_data.pq`
    2. Run `kmeans_pca.py` to generate the clustering results and store in csv files
       - set `method = 'KMeans'` or `method = 'KMedoids'` for different models
    3. run `backtesting.py` to implement the backtesting and get Brier scores
       - set `method = 'KMeans'` or `method = 'KMedoids'` for different models

### Problems

    If using `KMedoids` from `banditpam`, 
        - we need to wrap the original function for the sklearn-used ability (failed in `kmedoids_wrapper.py`)
        - also change the section in config file:
            ```
                MODEL_PARAM = {
                    'PCA': {'n_components': 0.95},
                    'KMeans': {'n_clusters': 20},
                    'KMedoids': {'n_clusters': 20},
                    # 'KMedoids': {'n_medois': 20},
                }
            ```
