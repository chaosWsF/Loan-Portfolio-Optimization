# Loan-Portfolio-Optimization

## Clustering Pipeline
    
1. Provide the settings of the experiment in `config.py`
2. Run `generating_raw_data.py` first to get cleaned `loans_data.pq`
   - Use `data_preview.ipynb` to take a glance at dataset
3. Run `kmeans_pca.py` to generate the clustering results and store in csv files
    - set `method = 'KMeans'` or `method = 'KMedoids'` for different models
4. run `backtesting.py` to implement the backtesting and get Brier scores
    - set `method = 'KMeans'` or `method = 'KMedoids'` for different models

### Problems

If using `KMedoids` from `banditpam`, 

- we need to wrap the original function for the sklearn-used ability (failed in `kmedoids_wrapper.py`)
- also uncomment the last line of the following section in the config file:
   ```
       MODEL_PARAM = {
           'PCA': {'n_components': 0.95},
           'KMeans': {'n_clusters': 20},
           'KMedoids': {'n_clusters': 20},
           # 'KMedoids': {'n_medois': 20}
       }
   ```

## Analysis of Results

Use `cluster_analysis.ipynb`
