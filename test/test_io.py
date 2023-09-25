import pytest
import argparse
import os
import numpy as np
import rmsd
from clusttraj.io import (
    ClustOptions,
    check_positive,
    extant_file,
    configure_runtime,
    parse_args,
    save_clusters_config,
)


def test_ClustOptions(options_dict):
    clust_opt = ClustOptions(**options_dict)

    assert os.path.basename(clust_opt.out_clust_name) == "clusters.dat"
    assert len(clust_opt.reorder_excl) == 0
    assert isinstance(clust_opt.reorder_excl, np.ndarray)
    assert clust_opt.exclusions is False
    assert clust_opt.reorder_alg_name == "hungarian"
    assert clust_opt.reorder_alg is None
    assert clust_opt.reorder is False
    assert clust_opt.input_distmat is False
    assert clust_opt.distmat_name == "test/ref/test_distmat.npy"
    assert os.path.basename(clust_opt.summary_name) == "clusters.out"
    assert os.path.basename(clust_opt.evo_name) == "clusters_evo.pdf"
    assert os.path.basename(clust_opt.dendrogram_name) == "clusters_dendrogram.pdf"
    assert os.path.basename(clust_opt.out_conf_name) == "clusters_confs"
    assert clust_opt.out_conf_fmt == "xyz"
    assert os.path.basename(clust_opt.mds_name) == "clusters.pdf"
    assert clust_opt.save_confs is False
    assert clust_opt.plot is False
    assert clust_opt.trajfile == "test/ref/testtraj.xyz"
    assert clust_opt.min_rmsd == 1.0
    assert clust_opt.method == "ward"
    assert clust_opt.n_workers == 1
    assert clust_opt.no_hydrogen is True
    assert clust_opt.opt_order is False
    assert clust_opt.solute_natoms == 17
    assert clust_opt.overwrite is True
    assert clust_opt.final_kabsch is False
    assert clust_opt.silhouette_score is False


def test_check_positive():
    assert check_positive("1") == 1

    with pytest.raises(ValueError):
        check_positive("a")

    with pytest.raises(argparse.ArgumentTypeError):
        check_positive("-1")


def test_extant_file():
    assert extant_file("test/ref/testtraj.xyz") == "test/ref/testtraj.xyz"

    with pytest.raises(argparse.ArgumentTypeError):
        extant_file("somewhere/nonexistent.file")


def test_parse_args():
    args = argparse.Namespace(
        natoms_solute=10,
        reorder_exclusions=[1, 2, 3],
        reorder_alg="hungarian",
        reorder=False,
        input=True,
        outputdistmat="distmat.npy",
        outputclusters="clusters.dat",
        clusters_configurations=True,
        plot=True,
        trajectory_file="trajectory.xyz",
        min_rmsd=0.1,
        method="ward",
        nprocesses=4,
        no_hydrogen=True,
        optimal_ordering=True,
        force=True,
        final_kabsch=True,
        silhouette_score=False,
    )
    clust_opt = parse_args(args)

    assert isinstance(clust_opt, ClustOptions)
    assert clust_opt.reorder_alg is None

    args = argparse.Namespace(
        natoms_solute=10,
        reorder_exclusions=[1, 2, 3],
        reorder_alg="hungarian",
        reorder=True,
        input=True,
        outputdistmat="distmat.npy",
        outputclusters="clusters.dat",
        clusters_configurations=True,
        plot=True,
        trajectory_file="trajectory.xyz",
        min_rmsd=0.1,
        method="ward",
        nprocesses=4,
        no_hydrogen=True,
        optimal_ordering=True,
        force=True,
        final_kabsch=True,
        silhouette_score=False,
    )
    clust_opt = parse_args(args)

    assert clust_opt.reorder_alg == rmsd.reorder_hungarian


def test_configure_runtime(caplog):
    clust_opt = configure_runtime(
        ["test/ref/testtraj.xyz", "--min-rmsd", "1.0", "-np", "1"]
    )

    assert clust_opt.trajfile == "test/ref/testtraj.xyz"
    assert clust_opt.min_rmsd == pytest.approx(1.0, abs=1e-8)
    assert clust_opt.n_workers == 1
    assert clust_opt.out_clust_name == "clusters.dat"

    with pytest.raises(SystemExit):
        clust_opt = configure_runtime(
            ["test/ref/testtraj.xyz", "--min-rmsd", "1.0", "-m", "nonexistent-method"]
        )

    with pytest.raises(SystemExit):
        clust_opt = configure_runtime(
            [
                "test/ref/testtraj.xyz",
                "--min-rmsd",
                "1.0",
                "--reorder-alg",
                "nonexistent-method",
            ]
        )

    with pytest.raises(SystemExit):
        clust_opt = configure_runtime(
            [
                "test/ref/testtraj.xyz",
                "--min-rmsd",
                "1.0",
                "-cc",
                "nonexistent-extension",
            ]
        )

    with pytest.raises(SystemExit):
        clust_opt = configure_runtime(
            ["test/ref/testtraj.xyz", "--min-rmsd", "1.0", "-n", "-eex", "1"]
        )


def test_save_clusters_config(clust_opt, clusters_seq, test_distmat):
    save_clusters_config(
        clust_opt.trajfile,
        clusters_seq,
        test_distmat,
        clust_opt.no_hydrogen,
        clust_opt.reorder_alg,
        clust_opt.solute_natoms,
        clust_opt.out_conf_name,
        clust_opt.out_conf_fmt,
        clust_opt.reorder_excl,
        clust_opt.final_kabsch,
        clust_opt.overwrite,
    )

    assert os.path.exists(clust_opt.out_conf_name + "_1." + clust_opt.out_conf_fmt)
    assert os.path.exists(clust_opt.out_conf_name + "_2." + clust_opt.out_conf_fmt)
    assert os.path.exists(clust_opt.out_conf_name + "_3." + clust_opt.out_conf_fmt)
