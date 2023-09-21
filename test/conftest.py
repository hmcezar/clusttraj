import pytest
import os
import numpy as np
from openbabel import pybel
from clusttraj.io import ClustOptions


@pytest.fixture
def pybel_mol():
    water_mol = (
        "3\n"
        "Water molecule\n"
        "O          0.00000        0.00000        0.11779\n"
        "H          0.00000        0.75545       -0.47116\n"
        "H          0.00000       -0.75545       -0.47116"
    )
    return pybel.readstring("xyz", water_mol)


@pytest.fixture
def first_conf_traj():
    return pybel.readfile("xyz", "test/ref/testtraj.xyz").__next__()


@pytest.fixture
def options_dict(tmp_path):
    options_dict = {
        "out_clust_name": os.path.join(tmp_path, "clusters.dat"),
        "reorder_excl": np.asarray([], np.int32),
        "exclusions": False,
        "reorder_alg_name": "hungarian",
        "reorder_alg": None,
        "reorder": False,
        "input_distmat": False,
        "distmat_name": "test/ref/test_distmat.npy",
        "summary_name": os.path.join(tmp_path, "clusters.out"),
        "evo_name": os.path.join(tmp_path, "clusters_evo.pdf"),
        "dendrogram_name": os.path.join(tmp_path, "clusters_dendrogram.pdf"),
        "out_conf_name": os.path.join(tmp_path, "clusters_confs"),
        "out_conf_fmt": "xyz",
        "mds_name": os.path.join(tmp_path, "clusters.pdf"),
        "save_confs": False,
        "plot": False,
        "trajfile": "test/ref/testtraj.xyz",
        "min_rmsd": 1.0,
        "method": "ward",
        "n_workers": 1,
        "no_hydrogen": True,
        "opt_order": False,
        "solute_natoms": 17,
        "overwrite": True,
        "final_kabsch": False,
        "silhouette_score": False,
    }

    return options_dict


@pytest.fixture
def clust_opt(options_dict):
    return ClustOptions(**options_dict)


@pytest.fixture
def test_distmat():
    return np.load("test/ref/test_distmat.npy")


@pytest.fixture
def clusters_seq():
    return np.array([1, 2, 3])


@pytest.fixture
def Z_matrix():
    return np.array([[0.0, 1.0, 2.94126393, 2.0], [2.0, 3.0, 4.6348889, 3.0]])
