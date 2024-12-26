"""Main entry point for clusttraj.

Can be called from command line or from an external library given a list
of arguments.
"""

import sys
import numpy as np
from typing import List
import time
from .io import Logger, configure_runtime, save_clusters_config
from .distmat import get_distmat
from .plot import plot_clust_evo, plot_dendrogram, plot_mds, plot_tsne
from .classify import classify_structures, classify_structures_silhouette
from .metrics import compute_metrics


def main(args: List[str] = None) -> None:
    """Main function that performs clustering and generates output.

    Args:
        args (List): List of command-line arguments. Defaults to None.

    Returns:
        None
    """
    global_start_time = time.monotonic()

    # parse command-line arguments
    if args is None:
        args = sys.argv[1:]

    # get ClustOptions class with parsed arguments
    clust_opt = configure_runtime(args)

    # get the distance matrix
    start_time = time.monotonic()
    distmat = get_distmat(clust_opt)
    end_time = time.monotonic()
    if clust_opt.verbose:
        Logger.logger.info(
            f"Time spent computing (or loading) distance matrix: {end_time - start_time:.6f} s\n"
        )

    # perform the clustering
    start_time = time.monotonic()
    if clust_opt.silhouette_score:
        Z, clusters = classify_structures_silhouette(clust_opt, distmat)
    else:
        Z, clusters = classify_structures(clust_opt, distmat)
    end_time = time.monotonic()
    if clust_opt.verbose:
        Logger.logger.info(f"Time spent clustering: {end_time - start_time:.6f} s\n")

    # get the elements closest to the centroid (see https://stackoverflow.com/a/39870085/3254658)
    if clust_opt.save_confs:
        start_time = time.monotonic()
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
            clust_opt.reorder_solvent_only,
            clust_opt.solute_natoms,
            clust_opt.weight_solute,
            clust_opt.out_conf_name,
            clust_opt.out_conf_fmt,
            clust_opt.reorder_excl,
            clust_opt.final_kabsch,
            clust_opt.overwrite,
        )
        end_time = time.monotonic()
        if clust_opt.verbose:
            Logger.logger.info(
                f"Time spent saving configurations: {end_time - start_time:.6f} s\n"
            )

    # generate plots
    if clust_opt.plot:
        start_time = time.monotonic()
        plot_clust_evo(clust_opt, clusters)

        plot_dendrogram(clust_opt, clusters, Z)

        plot_mds(clust_opt, clusters, distmat)

        plot_tsne(clust_opt, clusters, distmat)
        end_time = time.monotonic()
        if clust_opt.verbose:
            Logger.logger.info(f"Time spent plotting: {end_time - start_time:.6f} s\n")

    # print the cluster sizes
    outclust_str = f"A total {len(clusters)} snapshots were read and {max(clusters)} cluster(s) was(were) found.\n"
    outclust_str += "The cluster sizes are:\nCluster\tSize\n"

    labels, sizes = np.unique(clusters, return_counts=True)
    for label, size in zip(labels, sizes):
        outclust_str += f"{label}\t{size}\n"
    Logger.logger.info(outclust_str)

    # Compute the evaluation metrics
    if clust_opt.metrics:
        start_time = time.monotonic()
        ss, ch, db, cpcc = compute_metrics(distmat, Z, clusters)
        end_time = time.monotonic()

        outclust_str += f"\nSilhouette score: {ss:.3f}\n"
        outclust_str += f"Calinski Harabsz score: {ch:.3f}\n"
        outclust_str += f"Davies-Bouldin score: {db:.3f}\n"
        outclust_str += f"Cophenetic correlation coefficient: {cpcc:.3f}\n\n"

        if clust_opt.verbose:
            Logger.logger.info(
                f"Time spent computing metrics: {end_time - start_time:.6f} s\n"
            )

    # save summary
    with open(clust_opt.summary_name, "w") as f:
        f.write(str(clust_opt))
        f.write(outclust_str)

    global_end_time = time.monotonic()
    Logger.logger.info(f"Total wall time: {global_end_time - global_start_time:.6f} s\n")


if __name__ == "__main__":
    main()
