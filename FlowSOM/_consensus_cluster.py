import numpy as np
from itertools import combinations
import bisect
from joblib import Parallel, delayed
from sklearn.base import ClusterMixin

class ConsensusCluster:
    """
      Implementation of Consensus clustering, following the paper
      https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
      Args:
        * cluster -> clustering class
                    needs fit_predict method called with parameter n_clusters
        * L -> smallest number of clusters to try
        * K -> biggest number of clusters to try
        * H -> number of resamplings for each cluster number
        * resample_proportion -> percentage to sample
        * Mk -> consensus matrices for each k (shape =(K,data.shape[0],data.shape[0]))
                (NOTE: every consensus matrix is retained, like specified in the paper)
        * Ak -> area under CDF for each number of clusters 
                (see paper: section 3.3.1. Consensus distribution.)
        * deltaK -> changes in ares under CDF
                (see paper: section 3.3.1. Consensus distribution.)
        * self.bestK -> number of clusters that was found to be best
      """

    def __init__(self,
                 cluster: ClusterMixin,
                 L: int,
                 K: int,
                 H: int,
                 resample_proportion: int = 0.5,
                 random_state: int = 187):
        assert 0 <= resample_proportion <= 1, "proportion has to be between 0 and 1"
        self.cluster_ = cluster
        self.resample_proportion_ = resample_proportion
        self.L_ = L
        self.K_ = K
        self.H_ = H
        self.Mk = None
        self.Ak = None
        self.deltaK = None
        self.bestK = None
        self.random_state = random_state

    def _internal_resample(self,
                           data: np.ndarray,
                           proportion: float):
        """
        Args:
          * data -> (examples,attributes) format
          * proportion -> percentage to sample
        """
        np.random.seed(self.random_state)
        resampled_indices = np.random.choice(
            range(data.shape[0]), size=int(data.shape[0]*proportion), replace=False)
        return resampled_indices, data[resampled_indices, :]

    def _create_cluster_instance(self,
                                 k: int) -> ClusterMixin:
        try:
            return self.cluster_(n_clusters = k,
                                 random_state = self.random_state)
        except TypeError as e:
            if "got an unexpected keyword argument 'random_state'" in str(e):
                return self.cluster_(n_clusters = k)

    def _fit_cdf_area_differences(self,
                                  array: np.ndarray) -> np.ndarray: 
        hist, bins = np.histogram(array.ravel(), density=True)
        return np.sum(
            np.fromiter(
                (h*(b-a) for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist))),
                dtype = np.float32
            )
        )
 
    def _calculate_consensus_matrix(self,
                                    k: int,
                                    data: np.ndarray) -> np.ndarray:
        # sourcery skip: class-extract-method
        Mk = np.zeros((data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],)*2)
        for _ in range(self.H_):  # resample H times
            resampled_indices, resample_data = self._internal_resample(
                data, self.resample_proportion_)
            clf = self._create_cluster_instance(k)
            Mh = clf.fit_predict(resample_data)
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
                    Mk[ids_[0], ids_[1]] += 1
            # increment counts
            ids_2 = np.array(list(combinations(resampled_indices, 2))).T
            Is[ids_2[0], ids_2[1]] += 1
        Mk /= Is+1e-8  # consensus matrix
        # Mk[i_] is upper triangular (with zeros on diagonal), we now make it symmetric
        Mk += Mk.T
        Mk[range(data.shape[0]), range(
            data.shape[0])] = 1  # always with self
        return Mk
    
    def fit(self,
            data: np.ndarray,
            n_jobs: int) -> None:
        """
        Fits a consensus matrix for each number of clusters, uses joblib to parallelize
        Args:
          * data -> (examples,attributes) format
          * verbose -> should print or not
        """
        consensus_matrices: list[np.ndarray] = Parallel(n_jobs = n_jobs)(
            delayed(self._calculate_consensus_matrix)
            (k, data) for k in range(self.L_, self.K_)
        )
        self.Mk = np.array(consensus_matrices)
        self.Ak = Parallel(n_jobs = n_jobs)(
            delayed(self._fit_cdf_area_differences)
            (array) for array in self.Mk
        )
        self.deltaK = np.array([(Ab-Aa)/Aa if i > 2 else Aa
                                for Ab, Aa, i in zip(self.Ak[1:], self.Ak[:-1], range(self.L_, self.K_-1))])
        self.bestK = np.argmax(self.deltaK) + \
            self.L_ if self.deltaK.size > 0 else self.L_

    def predict(self) -> np.ndarray:
        """
        Predicts on the consensus matrix, for best found cluster number
        """
        assert self.Mk is not None, "First run fit"
        clf = self._create_cluster_instance(self.bestK)
        return clf.fit_predict(1-self.Mk[self.bestK-self.L_])

    def predict_data(self, data) -> np.ndarray:
        """
        Predicts on the data, for best found cluster number
        Args:
          * data -> (examples,attributes) format 
        """
        assert self.Mk is not None, "First run fit"
        clf = self._create_cluster_instance(self.bestK)
        return clf.fit_predict(data)