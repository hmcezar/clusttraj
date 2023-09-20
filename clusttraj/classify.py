import scipy.cluster.hierarchy as hcl
from scipy.spatial.distance import squareform
from sklearn import metrics
import numpy as np
from typing import Tuple
from .io import ClustOptions, Logger


def silh_score(X, Z, dstep=0.1):
    """
    Find the optimal threshold following the silhouette score metric.

    :param X: Array with initial data (distmat matrix)
    :type X: numpy.ndarray

    :param Z: 'Z' matrix from the scipy.cluster.hierarchy.linkage() method
    :type Z: numpy.ndarray

    :param dstep: Interval between threshold values, defaults to 0.1
    :type dstep: float

    :return: The optimal silhouette score value 
             :rtype: numpy.float64

             An array with degenerated threshold values that yield the same optimal score
             :rtype: numpy.ndarray

             An array with the cluster's labels from each optimal score
             :rtype: numpy.ndarray
    """
    
    # Initialize the score and threshold
    ss_opt = np.float64()
    t_opt = np.array([])
    labels_opt = np.array([])

    # Get the range of threshold values
    t_range = np.arange(min(Z[:, 2]), max(Z[:, 2]), dstep)

    for t in t_range:
        # Create an array with cluster labels
        hcl_labels = hcl.fcluster(Z, t=t, criterion='distance')
        
        # Compute the silhouette score
        ss = metrics.silhouette_score(X, hcl_labels, metric='precomputed')

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

    return np.round(ss_opt, 3), np.round(t_opt, 3), labels_opt


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

    if clust_opt.silhouette_score:
        ss_opt, t_opt, labels_opt = silh_score(squareform(distmat), Z)

        print("Highest silhouette score: %s" % ss_opt)

        if t_opt.size > 1:
            Logger.logger.info("The following RMSD threshold values yielded the same optimial silhouette score: %s" % t_opt)
            Logger.logger.info("Clusters labels for each threshold: %s" % labels_opt)
            Logger.logger.info("The smallest RMSD of {} has been adopted with the corresponding labels: {}".format(t_opt[0], labels_opt[0]))
            clusters = labels_opt[0]

        else:
            Logger.logger.info("Optimal RMSD threshold value: %s" % t_opt)
            clusters = labels_opt
        
        return Z, clusters, t_opt

    else:
        # build the clusters
        clusters = hcl.fcluster(Z, clust_opt.min_rmsd, criterion='distance')

        Logger.logger.info(
            f"Saving clustering classification to {clust_opt.out_clust_name}\n"
        )
        np.savetxt(clust_opt.out_clust_name, clusters, fmt="%d")

        return Z, clusters
