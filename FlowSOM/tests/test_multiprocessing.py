import numpy as np
from itertools import combinations
import bisect
from sklearn.base import ClusterMixin
import pytest

from .._consensus_cluster import ConsensusCluster


class original_ConsensusCluster(ConsensusCluster):
    """
    original fit function of https://github.com/ZigaSajovic/Consensus_Clustering
    """
    def __init__(self,
                 cluster: ClusterMixin,
                 L: int,
                 K: int,
                 H: int,
                 resample_proportion: int = 0.5,
                 random_state: int = 187):
        super().__init__(cluster = cluster,
                         L = L,
                         K = K,
                         H = H,
                         resample_proportion = resample_proportion,
                         random_state = random_state)
        
    def fit(self, data, verbose=False) -> None:
        """
        Fits a consensus matrix for each number of clusters
        Args:
          * data -> (examples,attributes) format
          * verbose -> should print or not
        """
        Mk = np.zeros((self.K_-self.L_, data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],)*2)
        for k in range(self.L_, self.K_):  # for each number of clusters
            i_ = k-self.L_
            if verbose:
                print("At k = %d, aka. iteration = %d" % (k, i_))
            for h in range(self.H_):  # resample H times
                if verbose:
                    print("\tAt resampling h = %d, (k = %d)" % (h, k))
                resampled_indices, resample_data = self._internal_resample(
                    data, self.resample_proportion_)
                Mh = self.cluster_(n_clusters=k,
                                   random_state = self.random_state).fit_predict(resample_data)
                # find indexes of elements from same clusters with bisection
                # on sorted array => this is more efficient than brute force search
                id_clusts = np.argsort(Mh)
                sorted_ = Mh[id_clusts]
                for i in range(k):  # for each cluster
                    ia = bisect.bisect_left(sorted_, i)
                    ib = bisect.bisect_right(sorted_, i)
                    is_ = id_clusts[ia:ib]
                    ids_ = np.array(list(combinations(is_, 2))).T
                    # sometimes only one element is in a cluster (no combinations)
                    if ids_.size != 0:
                        Mk[i_, ids_[0], ids_[1]] += 1
                # increment counts
                ids_2 = np.array(list(combinations(resampled_indices, 2))).T
                Is[ids_2[0], ids_2[1]] += 1
            Mk[i_] /= Is+1e-8  # consensus matrix
            # Mk[i_] is upper triangular (with zeros on diagonal), we now make it symmetric
            Mk[i_] += Mk[i_].T
            Mk[i_, range(data.shape[0]), range(
                data.shape[0])] = 1  # always with self
            Is.fill(0)  # reset counter
        self.Mk = Mk
        # fits areas under the CDFs
        self.Ak = np.zeros(self.K_-self.L_)
        for i, m in enumerate(Mk):
            hist, bins = np.histogram(m.ravel(), density=True)
            self.Ak[i] = np.sum(h*(b-a)
                             for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
        # fits differences between areas under CDFs
        self.deltaK = np.array([(Ab-Aa)/Aa if i > 2 else Aa
                                for Ab, Aa, i in zip(self.Ak[1:], self.Ak[:-1], range(self.L_, self.K_-1))])
        self.bestK = np.argmax(self.deltaK) + \
            self.L_ if self.deltaK.size > 0 else self.L_

def test_implemented_classifiers():
    from ..cluster import cluster
    data_array = np.random.randn(100,15)
    with pytest.raises(NotImplementedError):
        cluster(data_array,
                x_dim = 50,
                y_dim = 50,
                consensus_cluster_algorithm = "somthing")

def test_multiprocessing():
    from minisom import MiniSom
    from ..cluster import fetch_winning_cluster
    from joblib import Parallel, delayed
    from sklearn.cluster import KMeans
    x_dim = 10
    y_dim = 10
    data = np.random.randn(100, 15)
    n_features = data.shape[1]

    som = MiniSom(x = x_dim,
                  y = y_dim,
                  input_len = n_features,
                  sigma = 1,
                  learning_rate = 0.5,
                  neighborhood_function = "gaussian",
                  random_seed = 187
                 )
    som.pca_weights_init(data)
    som.train(data,
              num_iteration = 100,
              verbose = False)
    
    weights = som.get_weights()

    flattened_weights = weights.reshape(x_dim*y_dim,
                                        n_features)
    
    cluster_ = ConsensusCluster(KMeans,
                                10,
                                20,
                                10,
                                resample_proportion = 0.7,
                                random_state = 187)
    cluster_.fit(flattened_weights,
                 n_jobs = 2)
    flattened_classes = cluster_.predict_data(flattened_weights)
    
    map_class = flattened_classes.reshape(x_dim, y_dim)

    data_parallel = Parallel(n_jobs = 2)(
        delayed(fetch_winning_cluster)
        (som, data[i,:], map_class) for i in range(data.shape[0])
    )

    cluster_ = original_ConsensusCluster(KMeans,
                                         10,
                                         20,
                                         10,
                                         resample_proportion = 0.7,
                                         random_state = 187)
    cluster_.fit(flattened_weights)
    flattened_classes = cluster_.predict_data(flattened_weights)
    map_class = flattened_classes.reshape(x_dim, y_dim)
    data_original = Parallel(n_jobs = 2)(
        delayed(fetch_winning_cluster)
        (som, data[i,:], map_class) for i in range(data.shape[0])
    )
    assert data_original == data_parallel