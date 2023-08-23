import scipy.cluster.hierarchy as hcl
import numpy as np
from .io import Logger


def classify_structures(clust_opt, distmat):
    # linkage
    Logger.logger.info(
        f"Starting clustering using '{clust_opt.method}' method to join the clusters\n"
    )
    Z = hcl.linkage(distmat, clust_opt.method, optimal_ordering=clust_opt.opt_order)

    # build the clusters and print them to file
    clusters = hcl.fcluster(Z, clust_opt.min_rmsd, criterion="distance")
    Logger.logger.info(
        f"Saving clustering classification to {clust_opt.out_clust_name}\n"
    )
    np.savetxt(clust_opt.out_clust_name, clusters, fmt="%d")

    return Z, clusters
