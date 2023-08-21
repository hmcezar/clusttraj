"""
This script takes a trajectory and based on a minimal RMSD classify the structures in clusters.

The RMSD implementation using the Kabsch algorithm to superpose the molecules is taken from: https://github.com/charnley/rmsd
A very good description of the problem of superposition can be found at http://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures
A very good tutorial on hierachical clustering with scipy can be seen at https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
This script performs agglomerative clustering as suggested in https://stackoverflow.com/questions/31085393/hierarchical-clustering-a-pairwise-distance-matrix-of-precomputed-distances

Author: Henrique Musseli Cezar
Date: NOV/2017


TODO: 
    - [ ] split this file into files (compute distance, cluster, plot, etc..)
    - [ ] add unit tests for the routines
    - [ ] support coverage
    - [x] check why clusttraj is not being made available when I pip install
    - [ ] create docker to check the installation in a completely clean env
    - [ ] create conda package
    - [ ] update readme (also include installation instructions)
    - [ ] upload package
"""

import sys
import numpy as np
import scipy.cluster.hierarchy as hcl
from scipy.spatial.distance import squareform
from sklearn import manifold
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
from .io import configure_runtime, save_clusters_config
from .distmat import get_distmat


def main():
    clust_opt = configure_runtime(sys.argv[1:])

    distmat = get_distmat(clust_opt)

    # linkage
    logging.info(f"Starting clustering using '{clust_opt.method}' method to join the clusters\n")
    Z = hcl.linkage(distmat, clust_opt.method, optimal_ordering=clust_opt.opt_order)

    # build the clusters and print them to file
    clusters = hcl.fcluster(Z, clust_opt.min_rmsd, criterion='distance')
    logging.info(f"Saving clustering classification to {clust_opt.out_clust_name}\n")
    np.savetxt(clust_opt.out_clust_name, clusters, fmt='%d')
    
    # get the elements closest to the centroid (see https://stackoverflow.com/a/39870085/3254658)
    if clust_opt.save_confs:
        outconf = clust_opt.out_conf_name+"_*."+clust_opt.out_conf_fmt
        logging.info(f"Writing superposed configurations per cluster to files {outconf}\n")
        save_clusters_config(clust_opt.trajfile, clusters, distmat, clust_opt.no_hydrogen, clust_opt.reorder_alg, clust_opt.solute_natoms, clust_opt.out_conf_name, clust_opt.out_conf_fmt, clust_opt.reorder_excl)

    if clust_opt.plot:
        # plot evolution with o cluster in trajectory
        plt.figure(figsize=(25, 10))
        plt.plot(range(1,len(clusters)+1), clusters, "o-", markersize=4)
        plt.xlabel('Sample Index')
        plt.ylabel('Cluster classification')
        plt.savefig(clust_opt.evo_name, bbox_inches='tight')

        # plot the dendrogram
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('RMSD')
        hcl.dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
        )
        plt.axhline(args.min_rmsd,linestyle='--')
        plt.savefig(clust_opt.dendrogram_name, bbox_inches='tight')

        # finds the 2D representation of the distance matrix (multidimensional scaling) and plot it
        plt.figure()
        mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=666, n_init=3, max_iter=200, eps=1e-3, n_jobs=clust_opt.n_workers)
        coords = mds.fit_transform(squareform(distmat))
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.scatter(coords[:, 0], coords[:, 1], marker = 'o', c=clusters, cmap=plt.cm.nipy_spectral)
        plt.savefig(clust_opt.mds_name, bbox_inches='tight')


    # print the cluster sizes
    outclust_str = f"A total {len(clusters)} snapshots were read and {max(clusters)} cluster(s) was(were) found.\n" 
    outclust_str += f"The cluster sizes are:\nCluster\tSize\n"

    labels, sizes = np.unique(clusters, return_counts=True)
    for label, size in zip(labels,sizes):
        outclust_str += f"{label}\t{size}\n"
    logging.info(outclust_str)

    # save summary
    with open(clust_opt.summary_name, "w") as f:
        f.write(str(clust_opt))
        f.write(outclust_str)

if __name__ == '__main__':
    main()
