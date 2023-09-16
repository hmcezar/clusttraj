"""
This script takes a trajectory and based on a minimal RMSD classify the structures in clusters.

The RMSD implementation using the Kabsch algorithm to superpose the molecules is taken from: https://github.com/charnley/rmsd
A very good description of the problem of superposition can be found at http://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures
A very good tutorial on hierachical clustering with scipy can be seen at https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
This script performs agglomerative clustering as suggested in https://stackoverflow.com/questions/31085393/hierarchical-clustering-a-pairwise-distance-matrix-of-precomputed-distances

Author: Henrique Musseli Cezar
Date: NOV/2017


TODO: 
    - [x] split this file into files (compute distance, cluster, plot, etc..)
    - [x] add unit tests for the routines
    - [ ] support coverage
    - [x] check why clusttraj is not being made available when I pip install
    - [ ] create conda package
    - [x] update readme (also include installation instructions)
    - [ ] upload package
"""

import sys
import numpy as np
from .io import Logger, configure_runtime, save_clusters_config
from .distmat import get_distmat
from .plot import plot_clust_evo, plot_dendrogram, plot_mds
from .classify import classify_structures


def main(args=None) -> None:
    """
    Main function that performs clustering and generates output.

    Args:
        args (list): List of command-line arguments. Defaults to None.

    Returns:
        None
    """
    if args is None:
        args = sys.argv[1:]

    # get ClustOptions class with parsed arguments
    clust_opt = configure_runtime(args)

    # get the distance matrix
    distmat = get_distmat(clust_opt)

    # perform the actual clustering
    Z, clusters = classify_structures(clust_opt, distmat)

    # get the elements closest to the centroid (see https://stackoverflow.com/a/39870085/3254658)
    if clust_opt.save_confs:
        outconf = clust_opt.out_conf_name + "_*." + clust_opt.out_conf_fmt
        Logger.logger.info(
            f"Writing superposed configurations per cluster to files {outconf}\n"
        )
        save_clusters_config(
            clust_opt.trajfile,
            clusters,
            distmat,
            clust_opt.no_hydrogen,
            clust_opt.reorder_alg,
            clust_opt.solute_natoms,
            clust_opt.out_conf_name,
            clust_opt.out_conf_fmt,
            clust_opt.reorder_excl,
            clust_opt.final_kabsch,
            clust_opt.overwrite,
        )

    # generate plots
    if clust_opt.plot:
        plot_clust_evo(clust_opt, clusters)

        plot_dendrogram(clust_opt, Z)

        plot_mds(clust_opt, clusters, distmat)

    # print the cluster sizes
    outclust_str = f"A total {len(clusters)} snapshots were read and {max(clusters)} cluster(s) was(were) found.\n"
    outclust_str += f"The cluster sizes are:\nCluster\tSize\n"

    labels, sizes = np.unique(clusters, return_counts=True)
    for label, size in zip(labels, sizes):
        outclust_str += f"{label}\t{size}\n"
    Logger.logger.info(outclust_str)

    # save summary
    with open(clust_opt.summary_name, "w") as f:
        f.write(str(clust_opt))
        f.write(outclust_str)


if __name__ == "__main__":
    main()
