import os
from clusttraj.plot import plot_clust_evo, plot_dendrogram, plot_mds


def test_plot_clust_evo(clust_opt, clusters_seq):
    assert plot_clust_evo(clust_opt, clusters_seq) is None

    assert os.path.exists(clust_opt.evo_name)


def test_plot_dendrogram(clust_opt, Z_matrix):
    assert plot_dendrogram(clust_opt, Z_matrix) is None

    assert os.path.exists(clust_opt.dendrogram_name)


def test_plot_mds(clust_opt, clusters_seq, test_distmat):
    assert plot_mds(clust_opt, clusters_seq, test_distmat) is None

    assert os.path.exists(clust_opt.mds_name)
