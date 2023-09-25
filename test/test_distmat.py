import pytest
from clusttraj.utils import get_mol_info
from clusttraj.distmat import build_distance_matrix, get_distmat, compute_distmat_line
from clusttraj.io import ClustOptions


def test_get_distmat(options_dict, test_distmat):
    distmat = get_distmat(ClustOptions(**options_dict))
    assert len(distmat) == 3
    assert distmat == pytest.approx(test_distmat, abs=1e-8)

    options_dict["input_distmat"] = True
    distmat = get_distmat(ClustOptions(**options_dict))
    assert len(distmat) == 3
    assert distmat == pytest.approx(test_distmat, abs=1e-8)


def test_compute_distmat_line(options_dict, clust_opt, first_conf_traj, test_distmat):
    print(options_dict)
    line = compute_distmat_line(
        0,
        get_mol_info(first_conf_traj),
        clust_opt.trajfile,
        clust_opt.no_hydrogen,
        None,
        clust_opt.solute_natoms,
        clust_opt.reorder_excl,
        clust_opt.final_kabsch,
    )

    print(line)
    print(test_distmat)
    assert len(line) == 2
    assert line[0] == pytest.approx(test_distmat[0], abs=1e-8)
    assert line[1] == pytest.approx(test_distmat[1], abs=1e-8)


def test_build_distance_matrix(clust_opt, test_distmat):
    distmat = build_distance_matrix(clust_opt)
    assert len(distmat) == 3
    assert distmat[0] == pytest.approx(test_distmat[0], abs=1e-8)
    assert distmat[1] == pytest.approx(test_distmat[1], abs=1e-8)
    assert distmat[2] == pytest.approx(test_distmat[2], abs=1e-8)
