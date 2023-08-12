from sklearn.cluster import (AgglomerativeClustering,
                             Birch,
                             FeatureAgglomeration,
                             KMeans,
                             BisectingKMeans,
                             MiniBatchKMeans,
                             SpectralClustering)

IMPLEMENTED_CLASSIFIERS = {
    "AgglomerativeClustering": AgglomerativeClustering,
    "Birch": Birch,
    "FeatureAgglomeration": FeatureAgglomeration,
    "KMeans": KMeans,
    "BisectingKMeans": BisectingKMeans,
    "MiniBatchKMeans": MiniBatchKMeans,
    "SpectralClustering": SpectralClustering
}