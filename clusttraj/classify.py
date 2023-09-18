import scipy.cluster.hierarchy as hcl
import numpy as np
from typing import Tuple
from .io import ClustOptions, Logger


def classify_structures(
    clust_opt: ClustOptions, distmat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify structures based on clustering options and distance matrix.

    Args:
        clust_opt: The clustering options.
        distmat: The distance matrix.

    Returns:
        A tuple containing the linkage matrix and the clusters.
    """
    # linkage
    Logger.logger.info(
        f"Clustering using '{clust_opt.method}' method to join the clusters\n"
    )
    Z = hcl.linkage(distmat, clust_opt.method, optimal_ordering=clust_opt.opt_order)

    # build the clusters and print them to file
    clusters = hcl.fcluster(Z, clust_opt.min_rmsd, criterion="distance")
    Logger.logger.info(
        f"Saving clustering classification to {clust_opt.out_clust_name}\n"
    )
    np.savetxt(clust_opt.out_clust_name, clusters, fmt="%d")

    return Z, clusters
