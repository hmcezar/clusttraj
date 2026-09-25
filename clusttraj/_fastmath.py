"""Vectorized math helpers (NumPy-only, no Cython).

These mirror the speedups from charnley/rmsd#133 (vectorized weighted
Kabsch) so clusttraj gets the same wins without depending on an unmerged
upstream release. Installed rmsd 1.6.5 still has the triple-loop
``kabsch_weighted``; this module provides a drop-in fast equivalent.

All functions preserve rmsd semantics; numerical differences vs the
reference triple-loop are at the ~1e-15 level (floating-point
re-association only).
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def hungarian_ref_groups(ref_atoms):
    """Precompute per-element index groups for the *reference* atom array.

    The reference side (``Qa[view]``) is identical for every pair of a matrix
    line, so its ``np.unique``/``np.where`` work is done once per line instead
    of once per pair. Memory cost is O(N) ints per line — negligible even for
    very large trajectories since it is line-local, not trajectory-wide.
    """
    ref_atoms = np.asarray(ref_atoms)
    unique = np.unique(ref_atoms)
    groups = {int(a): np.where(ref_atoms == a)[0] for a in unique}
    return unique, groups


def reorder_hungarian_refcached(p_atoms, q_atoms, p_coord, q_coord, ref_unique=None, ref_groups=None):
    """Exact equivalent of ``rmsd.reorder_hungarian`` with cached ref side.

    ``p_atoms``/``p_coord`` are the reference side (fixed per line):
    pass ``ref_unique, ref_groups`` from :func:`hungarian_ref_groups` to skip
    redoing its ``unique``/``where`` per pair. The moving side (``q_*``) is
    still grouped live, so varying topologies fall back safely to identical
    semantics (including the empty-group edge cases).
    """
    p_atoms = np.asarray(p_atoms)
    q_atoms = np.asarray(q_atoms)
    if ref_unique is None or ref_groups is None:
        ref_unique, ref_groups = hungarian_ref_groups(p_atoms)
    view_reorder = np.full(q_atoms.shape, -1, dtype=int)
    for atom in ref_unique:
        p_idx = ref_groups[int(atom)]
        (q_idx,) = np.where(q_atoms == atom)
        distances = cdist(p_coord[p_idx], q_coord[q_idx], "euclidean")
        _, winner_cols = linear_sum_assignment(distances)
        view_reorder[p_idx] = q_idx[winner_cols]
    return view_reorder


def kabsch_weighted_fast(P, Q, W=None):
    """Vectorized equivalent of ``rmsd.kabsch_weighted``.

    Returns (U, V, rmsd) with identical conventions:
        P' = P @ U + V
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    n = P.shape[0]
    if W is None:
        w1 = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w1 = np.asarray(W, dtype=np.float64).reshape(-1)
    wsum = w1.sum()
    # rmsd's `iw = 3 / W.sum()` where W was tiled to (N,3); == 1 / w1.sum()
    iw = 1.0 / wsum if wsum != 0 else 0.0

    PW = P * w1[:, None]
    QW = Q * w1[:, None]
    CMP = PW.sum(axis=0)
    CMQ = QW.sum(axis=0)
    # C[i,k] = sum_j P[j,i]*Q[j,k]*w[j]
    C = PW.T @ Q
    C = (C - np.outer(CMP, CMQ) * iw) * iw

    PSQ = float((P * P * w1[:, None]).sum() - float(CMP @ CMP) * iw)
    QSQ = float((Q * Q * w1[:, None]).sum() - float(CMQ @ CMQ) * iw)

    V_, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V_) * np.linalg.det(Wt)) < 0.0
    if d:
        S[-1] = -S[-1]
        V_[:, -1] = -V_[:, -1]
    U = V_ @ Wt
    msd = (PSQ + QSQ) * iw - 2.0 * float(S.sum())
    if msd < 0.0:
        msd = 0.0
    rmsd_val = float(np.sqrt(msd))
    Vout = (CMP - U @ CMQ) * iw
    return U, Vout, rmsd_val


def kabsch_weighted_rmsd_fast(P, Q, W=None):
    """Fast ``rmsd.kabsch_weighted_rmsd``."""
    _, _, r = kabsch_weighted_fast(P, Q, W)
    return r


def weighted_rmsd_no_kabsch(P, Q, W):
    """Weighted RMSD without final rotation: sqrt(sum_i w_i |P_i-Q_i|^2)."""
    diff = P - Q
    return float(np.sqrt(np.dot(W, np.einsum("ij,ij->i", diff, diff))))


def build_weight_vector(n_total, natoms, weight_solute):
    """Build solute-weighted vector W (sums to 1)."""
    W = np.zeros(n_total, dtype=np.float64)
    W[:natoms] = weight_solute / natoms
    W[natoms:] = (1.0 - weight_solute) / (n_total - natoms)
    return W


def scatter_reordered(n_total, view, exclusions, p_excl, pview_reordered):
    """Rebuild full array from reordered view + excluded block.

    Replaces the slow ``np.insert(..., [x - whereins.tolist().index(x) ...])``
    pattern with a single O(N) scatter. Equivalent semantics.
    """
    out = np.empty((n_total,) + pview_reordered.shape[1:], dtype=pview_reordered.dtype)
    out[exclusions] = p_excl
    out[view] = pview_reordered
    return out
