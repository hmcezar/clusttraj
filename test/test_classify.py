import pytest
from clusttraj.classify import classify_structures, classify_structures_nclusters, classify_structures_silhouette


def test_classify_structures_silhouette(clust_opt):
    import numpy as np
    # This distmat reliably yields a single optimal threshold, causing an IndexError
    distmat = np.array([0.8914448312585805, 1.016370717302439, 1.0770540172126273, 0.5262279763713829, 0.9056642470478309, 0.7088570464110961, 0.49162431604355217, 0.8800215058217346, 0.6122727896102328, 0.7668009091886758])
    classify_structures_silhouette(clust_opt, distmat, dstep=0.1)



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
