from minisom import MiniSom

import numpy as np
import pandas as pd

from .consensus_cluster import ConsensusCluster

from typing import Union, Literal

import time
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
            random_state: int = 187):
    
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    input_len = data.shape[0]
    start = time.time() 
    som = MiniSom(x = x_dim,
                  y = y_dim,
                  input_len = 0,
                  sigma = sigma,
                  learning_rate = learning_rate,
                  neighborhood_function = neighborhood_function,
                  random_seed = random_state
                 )
    som.pca_weights_init(data)
    som.train(data,
              num_iteration = n_iterations,
              verbose = verbose)
    print("Training took ", time.time()-start, " seconds")
    
    weights = som.get_weights()
    flattened_weights = weights.reshape(x_dim*y_dim,
                                        input_len)
    cluster_ = ConsensusCluster(consensus_cluster_algorithm,
                                consensus_cluster_min_n,
                                consensus_cluster_max_n,
                                consensus_cluster_n_resamples,
                                resample_proportion = consensus_cluster_resample_proportion)
    cluster_.fit(flattened_weights, verbose = True)
    flattened_class = cluster_.predict_data(flattened_weights)
    map_class = flattened_class.reshape(x_dim, y_dim)
    start = time.time()
    labels = [fetch_winning_cluster(som = som,
                                  data_entry = data[i, :],
                                  cluster_map = map_class) for i in range(len(data))]
    print("Labeling took ", time.time() - start, " seconds")
    return labels