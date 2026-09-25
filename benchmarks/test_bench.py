"""Benchmarks for clusttraj (optional).

Run with::

    pip install -e ".[bench]"
    pytest benchmarks -q

Uses the MOx/methanol trajectory (100 frames, 41 atoms). Variants cover
a fast single-line kernel, a small 10-frame subset for quick iteration,
and the full 100-frame matrix used in ``test_mox_methanol.py`` — in both
``preload=True`` (in-memory) and streaming modes.
``n_workers=1`` and single-threaded BLAS give stable numbers.
"""

import os

import numpy as np
import pytest
import rmsd

pytest.importorskip("pytest_benchmark")

from clusttraj.distmat import (
    build_distance_matrix,
    compute_distmat_line,
    load_trajectory_arrays,
)
from clusttraj.io import ClustOptions
from clusttraj.utils import get_mol_info
from openbabel import pybel

TRAJFILE = "test/data/mox_4methanol.xyz"


def _mox_options(tmp_path, n_workers=1, trajfile=TRAJFILE):
    return ClustOptions(
        trajfile=trajfile,
        min_rmsd=None,
        n_workers=n_workers,
        method="ward",
        reorder_alg_name="hungarian",
        reorder_alg=rmsd.reorder_hungarian,
        out_conf_fmt="xyz",
        reorder=True,
        reorder_solvent_only=True,
        exclusions=False,
        no_hydrogen=True,
        input_distmat=False,
        save_confs=False,
        save_medoids=False,
        plot=False,
        opt_order=False,
        overwrite=True,
        final_kabsch=False,
        silhouette_score=False,
        n_clusters=4,
        metrics=False,
        distmat_name=os.path.join(str(tmp_path), "distmat.npy"),
        out_clust_name=os.path.join(str(tmp_path), "clusters.dat"),
        evo_name=None,
        mds_name=None,
        dendrogram_name=None,
        out_conf_name=None,
        out_medoids_name=None,
        out_medoids_fmt=None,
        summary_name=os.path.join(str(tmp_path), "clusters.out"),
        solute_natoms=17,
        weight_solute=0.9,
        reorder_excl=np.asarray([], np.int32),
        optimal_cut=None,
        verbose=False,
        # MOx (100x41) is tiny: preload into memory for the benchmarks.
        preload=True,
    )


def _write_subset(trajfile, out, n_frames):
    with open(trajfile) as f:
        lines = f.readlines()
    i, kept, count = 0, [], 0
    while i < len(lines) and count < n_frames:
        nat = int(lines[i].strip().split()[0])
        kept.extend(lines[i : i + 2 + nat])
        i += 2 + nat
        count += 1
    with open(out, "w") as f:
        f.writelines(kept)
    return out


def test_bench_line(benchmark, tmp_path):
    """Single matrix line (idx1=0, 99 pairs): kernel + I/O overhead."""
    mols = list(pybel.readfile("xyz", TRAJFILE))
    q_info = get_mol_info(mols[0])
    excl = np.asarray([], np.int32)
    result = benchmark(
        compute_distmat_line,
        0,
        q_info,
        TRAJFILE,
        True,
        rmsd.reorder_hungarian,
        True,
        17,
        0.9,
        excl,
        False,
    )
    assert len(result) == 99


def test_bench_mox_10frames(benchmark, tmp_path):
    """10-frame subset (45 pairs): quick iteration benchmark."""
    sub = _write_subset(TRAJFILE, os.path.join(str(tmp_path), "mox10.xyz"), 10)
    opt = _mox_options(str(tmp_path), n_workers=1, trajfile=sub)
    distmat = benchmark(build_distance_matrix, opt)
    assert len(distmat) == 45


def test_bench_mox_full(benchmark, tmp_path):
    """Full 100-frame MOx matrix (4950 pairs), single worker, preloaded."""
    opt = _mox_options(str(tmp_path), n_workers=1)
    distmat = benchmark(build_distance_matrix, opt)
    assert len(distmat) == 4950


def test_bench_mox_full_streaming(benchmark, tmp_path):
    """Full 100-frame MOx matrix streamed from disk (default path)."""
    opt = _mox_options(str(tmp_path), n_workers=1)
    opt.preload = False
    distmat = benchmark(build_distance_matrix, opt)
    assert len(distmat) == 4950


def test_bench_preload_only(benchmark):
    """Trajectory parsing alone (I/O floor for the full matrix)."""
    benchmark(load_trajectory_arrays, TRAJFILE)
