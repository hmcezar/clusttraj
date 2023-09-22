"""Main entry point for clusttraj.

Can be called from command line or from an external library given a list
of arguments.
"""

import sys
import numpy as np
from typing import List
from .io import Logger, configure_runtime, save_clusters_config
from .distmat import get_distmat
from .plot import plot_clust_evo, plot_dendrogram, plot_mds
from .classify import classify_structures, classify_structures_silhouette


def main(args: List[str] = None) -> None:
    """Main function that performs clustering and generates output.

    Args:
        args (List): List of command-line arguments. Defaults to None.

    Returns:
        None
    """
    if args is None:
        args = sys.argv[1:]

    # get ClustOptions class with parsed arguments
    clust_opt = configure_runtime(args)

    # get the distance matrix
    distmat = get_distmat(clust_opt)

    # perform the clustering
    if clust_opt.silhouette_score:
        Z, clusters = classify_structures_silhouette(clust_opt, distmat)
    else:
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
    outclust_str += "The cluster sizes are:\nCluster\tSize\n"

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
