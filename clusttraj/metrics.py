"""Functions to compute evaluation metrics of the clustering procedure"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cophenet
from typing import Tuple
import numpy as np
from .io import ClustOptions


def compute_metrics(
	clust_opt: ClustOptions, distmat: np.ndarray, z_matrix: np.ndarray, clusters: np.ndarray
) -> Tuple[np.float64, np.float64, np.float64, np.float64]:
	"""Compute the Cophenetic Correlation Coefficient and Silhouette, 
	Calinski Harabasz and Davies-Bouldin scores.

	Args:
        clust_opt (ClustOptions): The clustering options.
        z_matrix (np.ndarray): The Z-matrix from hierarchical clustering procedure.

    Returns:
        None
	"""

	# Compute the silhouette score
	ss = silhouette_score(squareform(distmat), clusters, metric="precomputed")

	# Compute the Calinski Harabasz score

	ch = calinski_harabasz_score(squareform(distmat), clusters)

	# Compute the Davies-Bouldin score

	db = davies_bouldin_score(squareform(distmat), clusters)

	# Compute the cophenetic correlation coefficient 

	# cpcc_coef, a = cophenet(z_matrix)
	cpcc_coef = cophenet(z_matrix)[0]
	# cpcc_coef = result[0]

	return ss, ch, db, cpcc_coef

