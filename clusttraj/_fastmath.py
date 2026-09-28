"""Small NumPy helpers for the RMSD pipeline (no Cython).

Requires ``rmsd>=1.7.0``, which ships the vectorized weighted Kabsch
(charnley/rmsd#133) — clusttraj calls ``rmsd.kabsch_weighted`` /
``rmsd.kabsch_weighted_rmsd`` directly. What remains here are
clusttraj-side savings upstream does not provide:

- direct-placement rebuild of reordered structures (``scatter_reordered``),
- solute weight-vector construction (``build_weight_vector``),
- weighted RMSD without the final rotation (``weighted_rmsd_no_kabsch``),
- caching of the Hungarian reference side, which is fixed per matrix
  line (``hungarian_ref_groups`` / ``reorder_hungarian_refcached`` —
  exactly equivalent to ``rmsd.reorder_hungarian``).
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
