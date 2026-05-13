import pytest
from clusttraj.classify import classify_structures, classify_structures_nclusters


def test_classify_structures(clust_opt, test_distmat, Z_matrix, clusters_seq):
    Z, clusters = classify_structures(clust_opt, test_distmat)

    assert Z == pytest.approx(Z_matrix, abs=1e-8)
    assert clusters == pytest.approx(clusters_seq, abs=1e-8)


def test_classify_structures_nclusters_two_clusters(clust_opt, test_distmat, Z_matrix):
    clust_opt.n_clusters = 2

    Z, clusters = classify_structures_nclusters(clust_opt, test_distmat)

    assert Z == pytest.approx(Z_matrix, abs=1e-8)
    assert clusters == pytest.approx([1, 1, 2], abs=1e-8)


def test_classify_structures_nclusters_three_clusters(clust_opt, test_distmat, Z_matrix):
    clust_opt.n_clusters = 3

    Z, clusters = classify_structures_nclusters(clust_opt, test_distmat)

    assert Z == pytest.approx(Z_matrix, abs=1e-8)
    assert clusters == pytest.approx([1, 2, 3], abs=1e-8)


def test_classify_structures_nclusters_too_many_clusters(clust_opt, test_distmat):
    clust_opt.n_clusters = 4

    with pytest.raises(ValueError, match="Cannot request 4 clusters"):
        classify_structures_nclusters(clust_opt, test_distmat)
