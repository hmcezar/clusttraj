"""Functions to perform clustering based on the RMSD matrix."""

import scipy.cluster.hierarchy as hcl
from scipy.spatial.distance import squareform
from sklearn import metrics
import numpy as np
from typing import Tuple
from .io import ClustOptions, Logger


def classify_structures_silhouette(
    clust_opt: ClustOptions, distmat: np.ndarray, dstep: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the optimal threshold following the silhouette score metric and
    perform the classification.

    Args:
        clust_opt: The clustering options.
        distmat: The RMSD matrix.
        dstep (float, optional): Interval between threshold values, defaults to 0.1

    Returns:
        A tuple containing the linkage matrix and the clusters.
    """

    # linkage
    Logger.logger.info(
        f"Clustering using '{clust_opt.method}' method to join the clusters\n"
    )
    Z = hcl.linkage(distmat, clust_opt.method, optimal_ordering=clust_opt.opt_order)

    # Initialize the score and threshold
    ss_opt = np.float64()
    t_opt = np.array([])
    labels_opt = np.array([])

    # Get the range of threshold values
    t_range = np.arange(min(Z[:, 2]), max(Z[:, 2]), dstep)

    for t in t_range:
        # Create an array with cluster labels
        hcl_labels = hcl.fcluster(Z, t=t, criterion="distance")

        # Compute the silhouette score
        ss = metrics.silhouette_score(
            squareform(distmat), hcl_labels, metric="precomputed"
        )

        # Check for degeneracy for the optimal threshold value
        if np.any(ss == ss_opt):
            ss_opt = ss
            t_opt = np.append(t_opt, t)
            labels_opt = np.vstack((labels_opt, hcl_labels))

        # Update the values to the highest silhouette score
        if np.all(ss > ss_opt):
            ss_opt = ss
            t_opt = t
            labels_opt = hcl_labels

    Logger.logger.info(f"Highest silhouette score: {ss_opt:.5f}\n")

    if t_opt.size > 1:
        t_opt_str = ", ".join([f"{t:.3f}" for t in t_opt])
        Logger.logger.info(
            f"The following RMSD threshold values yielded the same optimal silhouette score: {t_opt_str}\n"
        )
        Logger.logger.info(f"The smallest RMSD of {t_opt[0]:.5f} has been adopted\n")
        clusters = labels_opt[0]
    else:
        Logger.logger.info(f"Optimal RMSD threshold value: {t_opt[0]:.5f}\n")
        clusters = labels_opt

    clust_opt.update({"optimal_cut": t_opt})

    Logger.logger.info(
        f"Saving clustering classification to {clust_opt.out_clust_name}\n"
    )
    np.savetxt(clust_opt.out_clust_name, clusters, fmt="%d")

    return Z, clusters


def classify_structures(
    clust_opt: ClustOptions, distmat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify structures based on clustering options and RMSD matrix.

    Args:
        clust_opt: The clustering options.
        distmat: The RMSD matrix.

    Returns:
        A tuple containing the linkage matrix and the clusters.
    """
    # linkage
    Logger.logger.info(
        f"Clustering using '{clust_opt.method}' method to join the clusters\n"
    )
    Z = hcl.linkage(distmat, clust_opt.method, optimal_ordering=clust_opt.opt_order)

    # build the clusters
    clusters = hcl.fcluster(Z, clust_opt.min_rmsd, criterion="distance")

    Logger.logger.info(
        f"Saving clustering classification to {clust_opt.out_clust_name}\n"
    )
    np.savetxt(clust_opt.out_clust_name, clusters, fmt="%d")

    return Z, clusters


def find_medoids_from_clusters(distmat: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Find the medoids of the clusters.

    Args:
        distmat: The RMSD matrix.
        clusters: The clusters.

    Returns:
        The indices of the medoids.
    """
    n_clusters = len(np.unique(clusters))
    medoids = np.zeros(n_clusters, dtype=int)
    sq_distmat = squareform(distmat)

    for i in range(1, n_clusters + 1):
        indices = np.where(clusters == i)[0]
        distmat_cluster = sq_distmat[indices][:, indices]
        medoids[i - 1] = indices[np.argmin(np.sum(distmat_cluster, axis=0))]

    return medoids


def sum_distmat(distmat: np.ndarray) -> np.ndarray:
    """Sum the RMSD matrix.

    Args:
        distmat: The RMSD matrix.

    Returns:
        The sum of the RMSD matrix.
    """
    return np.sum(distmat)
