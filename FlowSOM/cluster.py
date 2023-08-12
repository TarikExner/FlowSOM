from typing import Union, Literal
import numpy as np
import pandas as pd
from minisom import MiniSom
from joblib import Parallel, delayed

from .consensus_cluster import ConsensusCluster
from ._cluster_algorithms import IMPLEMENTED_CLASSIFIERS

def fetch_winning_cluster(som: MiniSom,
                          data_entry: np.ndarray,
                          cluster_map: np.ndarray) -> int:
    winner = som.winner(data_entry)
    return cluster_map[winner]

def cluster(data: Union[np.ndarray, pd.DataFrame],
            x_dim: int,
            y_dim: int,
            sigma: float = 1,
            learning_rate: float = 0.5,
            n_iterations: int = 100,
            neighborhood_function = "gaussian",
            consensus_cluster_algorithm: Literal["AgglomerativeClustering"] = "AgglomerativeClustering",
            consensus_cluster_min_n: int = 10,
            consensus_cluster_max_n: int = 50,
            consensus_cluster_resample_proportion: float = 0.5,
            consensus_cluster_n_resamples: int = 10,
            verbose: bool = False,
            n_jobs: int = None,
            random_state: int = 187):
    
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    if consensus_cluster_algorithm not in IMPLEMENTED_CLASSIFIERS:
        raise NotImplementedError(f"Algorithm not implemented. Please choose from {list(IMPLEMENTED_CLASSIFIERS.keys())}")
    
    consensus_cluster_algorithm = IMPLEMENTED_CLASSIFIERS[consensus_cluster_algorithm]
    
    n_features = data.shape[1]

    som = MiniSom(x = x_dim,
                  y = y_dim,
                  input_len = n_features,
                  sigma = sigma,
                  learning_rate = learning_rate,
                  neighborhood_function = neighborhood_function,
                  random_seed = random_state
                 )
    som.pca_weights_init(data)
    som.train(data,
              num_iteration = n_iterations,
              verbose = verbose)
    
    weights = som.get_weights()
    flattened_weights = weights.reshape(x_dim*y_dim,
                                        n_features)
    cluster_ = ConsensusCluster(consensus_cluster_algorithm,
                                consensus_cluster_min_n,
                                consensus_cluster_max_n,
                                consensus_cluster_n_resamples,
                                resample_proportion = consensus_cluster_resample_proportion,
                                random_state = random_state)
    cluster_.fit(flattened_weights,
                 n_jobs = n_jobs)
    flattened_classes = cluster_.predict_data(flattened_weights)
    map_class = flattened_classes.reshape(x_dim, y_dim)

    return Parallel(n_jobs = n_jobs)(delayed(fetch_winning_cluster)(som, data[i,:], map_class) for i in range(data.shape[0]))