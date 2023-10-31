# FlowSOM

Python implementation of the FlowSOM algorithm based on [minisom](https://github.com/JustGlowing/minisom) and [ConsensusClustering](https://github.com/ZigaSajovic/Consensus_Clustering).

The main advantage over the current implementations is the parallelized nature of the ConsensusClustering.

## Installation

```
    pip install flowsom-clustering

```

## Basic usage:

```python
import numpy as np
import pandas as pd
from FlowSOM import flowsom

# import your data via pandas or numpy

data = pd.DataFrame(data = np.random.rand(250,10),
                    columns = [str(i) for i in range(10)])

cluster_annotations = flowsom(data = data, # input data array
                              x_dim = 50, # x dimension of the self organized map
                              y_dim = 50, # y dimension of the self organized map
                              sigma = 1, # spread of the neighborhood function
                              learning_rate = 0.5, # initial learning rate
                              n_iterations = 100, # trains the map for 100 iterations
                              neighborhood_function = "gaussian", # defines the neighborhood function
                              consensus_cluster_algorithm = "AgglomerativeClustering", # runs Consensus Clustering with Agglomerative Clustering
                              consensus_cluster_min_n = 10, # minimum n_clusters to check for optimal clustering
                              consensus_cluster_max_n = 50, # maximum n_clusters to check for optimal clustering
                              consensus_cluster_resample_proportion = 0.5, # resample proportion for the consensus finding
                              consensus_cluster_n_resamples = 10, # number of resamples for consensus clustering
                              verbose = False, # whether to be verbose
                              n_jobs = None, # number of joblib.Parallel threads
                              random_state = 187 # seed for reproducibility of the results
                              )

data["flowsom_consensus"] = cluster_annotations

```
