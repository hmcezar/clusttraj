"""Regression test for MOx in methanol (paper case).

Covers the solvent-informed setup used in Ribeiro & Cezar, JCTC 2025
(MOx/methanol): solute-solvent split with solvent-only reordering,
hydrogens excluded, solute weight ~0.9 on the 17 solute atoms, Ward
linkage cut to 4 clusters.

The 4 clusters cleanly split the syn/anti conformers defined by the
O-C-C-C dihedral of the solute (0-based solute indices 3-2-1-0):
two pure-syn clusters and two pure-anti clusters. The test pins the
cluster sizes, the medoid indices and the syn/anti composition per
cluster (matched via medoid, so it is invariant to cluster-label
permutation) to catch regressions in the RMSD/reordering/clustering
pipeline.
"""

import os

import numpy as np
import rmsd

from clusttraj.classify import (
    classify_structures_nclusters,
    find_medoids_from_clusters,
)
from clusttraj.distmat import get_distmat
from clusttraj.io import ClustOptions

TRAJFILE = "test/data/mox_4methanol.xyz"

# O-C-C-C dihedral (full-xyz 0-based indices) defining syn vs anti.
DIH_IDX = (3, 2, 1, 0)
SYN_CUTOFF_DEG = 90.0

# Ground truth from the reference run (ward, n_clusters=4, ns=17,
# ws=0.9, reorder solvent-only, no hydrogens).
EXPECTED_SIZES_SORTED = [15, 19, 30, 36]
EXPECTED_MEDOIDS_SORTED = [53, 70, 82, 83]
# medoid -> (cluster size, n_syn, n_anti)
EXPECTED_BY_MEDOID = {
    82: (36, 0, 36),
    70: (19, 19, 0),
    53: (30, 30, 0),
    83: (15, 0, 15),
}


def _read_xyz_coords(trajfile):
    """Parse an xyz trajectory into a list of (N, 3) arrays."""
    coords = []
    with open(trajfile) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        natoms = int(lines[i].strip().split()[0])
        frame = []
        for j in range(natoms):
            parts = lines[i + 2 + j].split()
            frame.append([float(x) for x in parts[1:4]])
        coords.append(np.asarray(frame))
        i += 2 + natoms
    return coords


def _dihedral_deg(p0, p1, p2, p3):
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def _syn_mask(trajfile):
    coords = _read_xyz_coords(trajfile)
    a, b, c, d = DIH_IDX
    diheds = np.array([_dihedral_deg(f[a], f[b], f[c], f[d]) for f in coords])
    return np.abs(diheds) < SYN_CUTOFF_DEG, diheds


def _mox_options(tmp_path):
    return ClustOptions(
        trajfile=TRAJFILE,
        min_rmsd=None,
        n_workers=1,
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
        distmat_name=os.path.join(tmp_path, "distmat.npy"),
        out_clust_name=os.path.join(tmp_path, "clusters.dat"),
        evo_name=None,
        mds_name=None,
        dendrogram_name=None,
        out_conf_name=None,
        out_medoids_name=None,
        out_medoids_fmt=None,
        summary_name=os.path.join(tmp_path, "clusters.out"),
        solute_natoms=17,
        weight_solute=0.9,
        reorder_excl=np.asarray([], np.int32),
        optimal_cut=None,
        verbose=False,
    )


def test_mox_methanol_4clusters_syn_anti(tmp_path):
    clust_opt = _mox_options(str(tmp_path))

    distmat = get_distmat(clust_opt)
    # 100 frames -> 100*99/2 condensed entries
    assert len(distmat) == 4950

    _, clusters = classify_structures_nclusters(clust_opt, distmat)
    assert len(clusters) == 100
    assert len(np.unique(clusters)) == 4

    sizes_sorted = sorted(int(np.sum(clusters == k)) for k in np.unique(clusters))
    assert sizes_sorted == EXPECTED_SIZES_SORTED

    medoids = find_medoids_from_clusters(distmat, clusters)
    assert sorted(medoids.tolist()) == EXPECTED_MEDOIDS_SORTED

    is_syn, _ = _syn_mask(TRAJFILE)
    # Trajectory itself is balanced syn/anti with a wide gap (max |dih|
    # syn ~47 deg, min |dih| anti ~153 deg), so the 90 deg cutoff is robust.
    assert int(is_syn.sum()) == 49
    assert int((~is_syn).sum()) == 51

    for medoid, (exp_size, exp_syn, exp_anti) in EXPECTED_BY_MEDOID.items():
        label = clusters[medoid]
        members = np.where(clusters == label)[0]
        assert len(members) == exp_size
        n_syn = int(is_syn[members].sum())
        n_anti = len(members) - n_syn
        assert (n_syn, n_anti) == (exp_syn, exp_anti)
        # Medoid conformer agrees with its cluster.
        assert bool(is_syn[medoid]) == (exp_syn > 0)

    # Every cluster is pure syn or pure anti (no mixing).
    for label in np.unique(clusters):
        members = np.where(clusters == label)[0]
        assert int(is_syn[members].sum()) in (0, len(members))
