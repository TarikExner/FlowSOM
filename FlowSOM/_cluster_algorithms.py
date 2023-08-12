from sklearn.cluster import (AgglomerativeClustering,
                             AffinityPropagation,
                             Birch,
                             DBSCAN,
                             FeatureAgglomeration,
                             KMeans,
                             BisectingKMeans,
                             MiniBatchKMeans,
                             MeanShift,
                             OPTICS,
                             SpectralClustering)

IMPLEMENTED_CLASSIFIERS = {
    "AgglomerativeClustering": AgglomerativeClustering,
    "AffinityPropagation": AffinityPropagation,
    "Birch": Birch,
    "DBSCAN": DBSCAN,
    "FeatureAgglomeration": FeatureAgglomeration,
    "KMeans": KMeans,
    "BisectingKMeans": BisectingKMeans,
    "MiniBatchKMeans": MiniBatchKMeans,
    "MeanShift": MeanShift,
    "OPTICS": OPTICS,
    "SpectralClustering": SpectralClustering
}