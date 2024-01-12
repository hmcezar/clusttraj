"""Functions to compute evaluation metrics of the clustering procedure"""

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cophenet
from typing import Tuple
import numpy as np
from .io import ClustOptions


def compute_metrics(
    clust_opt: ClustOptions,
    distmat: np.ndarray,
    z_matrix: np.ndarray,
    clusters: np.ndarray,
) -> Tuple[np.float64, np.float64, np.float64, np.float64]:
    """Compute metrics to assess the performance of the clustering procedure.

        Args:
        clust_opt (ClustOptions): The clustering options.
        z_matrix (np.ndarray): The Z-matrix from hierarchical clustering procedure.

    Returns:
        ss (np.float64): The silhouette score.
        ch (np.float64): The Calinski Harabasz score.
        db (np.float64): The Davies-Bouldin score.
        cpcc (np.float64): The cophenetic correlation coefficient.
    """

    # Compute the silhouette score
    ss = silhouette_score(squareform(distmat), clusters, metric="precomputed")

    # Compute the Calinski Harabasz score
    ch = calinski_harabasz_score(squareform(distmat), clusters)

    # Compute the Davies-Bouldin score
    db = davies_bouldin_score(squareform(distmat), clusters)

    # Compute the cophenetic correlation coefficient
    cpcc = cophenet(z_matrix)[0]

    return ss, ch, db, cpcc
