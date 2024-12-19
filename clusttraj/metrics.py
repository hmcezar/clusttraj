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


def compute_metrics(
    distmat: np.ndarray,
    z_matrix: np.ndarray,
    clusters: np.ndarray,
) -> Tuple[np.float64, np.float64, np.float64, np.float64]:
    """Compute metrics to assess the performance of the clustering procedure.

        Args:
        distmat: The distance matrix.
        z_matrix (np.ndarray): The Z-matrix from hierarchical clustering procedure.
        clusters (np.ndarray): The cluster classifications for each sample.

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
    cpcc, _ = cophenet(z_matrix, distmat)

    return ss, ch, db, cpcc
