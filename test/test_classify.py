import pytest
from clusttraj.classify import classify_structures


def test_classify_structures(clust_opt, test_distmat, Z_matrix, clusters_seq):
    Z, clusters = classify_structures(clust_opt, test_distmat)

    assert Z == pytest.approx(Z_matrix, abs=1e-8)
    assert clusters == pytest.approx(clusters_seq, abs=1e-8)
