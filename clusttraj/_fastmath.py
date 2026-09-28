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
- the shared alignment kernel (``prepare_ref`` / ``filter_moving`` /
  ``align_core``): every superposition step, in one place, used both for
  the RMSD values (``distmat``) and for the structures written to disk
  (``io.align_mol``), so the two can never drift apart.
"""

import numpy as np
import rmsd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import NamedTuple, Optional


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


def prepare_ref(q_atoms, q_coords, noh, nsatoms):
    """Filter hydrogens and center the reference side (Q).

    The single source for reference preparation, used for RMSD matrix lines
    and for the medoid written by the save functions alike.

    Returns ``(Qa, Q, natoms, qcenter)`` with ``Q`` centered at the solute
    centroid (``nsatoms`` set) or the full centroid, and ``natoms`` the
    number of non-hydrogen solute atoms derived from the reference.
    """
    q_atoms = np.asarray(q_atoms)
    q_all = np.asarray(q_coords, dtype=np.float64)
    if nsatoms:
        if noh:
            q_mask = q_atoms != 1
            Qa = q_atoms[q_mask]
            Q = q_all[q_mask].astype(np.float64, copy=True)
            natoms = int(np.count_nonzero(q_atoms[:nsatoms] != 1))
        else:
            Qa = q_atoms
            Q = q_all.astype(np.float64, copy=True)
            natoms = int(nsatoms)
    elif noh:
        q_mask = q_atoms != 1
        Qa = q_atoms[q_mask]
        Q = q_all[q_mask].astype(np.float64, copy=True)
        natoms = 0
    else:
        Qa = q_atoms
        Q = q_all.astype(np.float64, copy=True)
        natoms = 0
    if nsatoms:
        qcenter = Q[:natoms].mean(axis=0) if len(Q) else 0.0
    else:
        qcenter = Q.mean(axis=0) if len(Q) else 0.0
    return Qa, Q - qcenter, natoms, qcenter


def filter_moving(p_atoms, p_coords, noh, nsatoms):
    """Filter hydrogens for the moving frame; returns ``(Pa, P)`` (fresh copies)."""
    p_atoms = np.asarray(p_atoms)
    p_coords = np.asarray(p_coords)
    if noh:
        p_mask = p_atoms != 1
        return p_atoms[p_mask], p_coords[p_mask].astype(np.float64, copy=True)
    return p_atoms, np.array(p_coords, dtype=np.float64, copy=True)


def _do_reorder(reorder, qa_v, pa_v, q_v, p_v, cache, key):
    """Reorder via rmsd, caching the reference-side index groups.

    The reference side (``qa_v``) is identical for every pair of a matrix
    line; when the reorder function is exactly ``rmsd.reorder_hungarian``,
    its ``unique``/``where`` work on that side is done once per line. The
    cache is line-local (a few int arrays), so memory stays flat. Custom
    reorder callables go through untouched.
    """
    if reorder is rmsd.reorder_hungarian:
        entry = cache.get(key)
        if entry is None:
            entry = hungarian_ref_groups(qa_v)
            cache[key] = entry
        return reorder_hungarian_refcached(qa_v, pa_v, q_v, p_v, entry[0], entry[1])
    return reorder(qa_v, pa_v, q_v, p_v)


class Aligned(NamedTuple):
    """One aligned pair, as produced by :func:`align_core`.

    ``Pr``/``Pa`` are the filtered, reordered moving coords/atoms *before*
    the final rotation; ``kind`` tells which final step applies: ``"none"``
    (no final rotation), ``"weighted"`` (rotate by ``R``, translate by ``T``;
    ``w_rmsd`` is the pair value) or ``"plain"`` (rotate by ``R``).
    ``full`` carries the same rigid transforms applied to the complete
    moving coordinates (input atom order kept), or ``None`` when not asked
    for. ``R``/``T`` come from the very call the matrix value is computed
    with, so written structures carry exactly the matrix superposition.
    """

    Pr: np.ndarray
    Pa: np.ndarray
    W: Optional[np.ndarray]
    R: Optional[np.ndarray]
    T: Optional[np.ndarray]
    kind: str
    w_rmsd: Optional[float]
    full: Optional[np.ndarray]


def align_core(
    P,
    Pa,
    Q,
    Qa,
    natoms,
    nsatoms,
    reorder,
    reorder_solvent_only,
    excl_arr,
    weight_solute,
    final_kabsch,
    cache=None,
    W=None,
    p_full=None,
):
    """Superpose one moving frame onto the reference; shared by distmat and io.

    Runs every alignment step in one place: centering, solute-first rotation,
    solute reorder (with re-rotation), solvent reorder, solute weights, and
    the final rotation. Neither input is mutated; ``Q`` must already be
    centered (see :func:`prepare_ref`). ``natoms`` is the reference-side
    non-hydrogen solute count. ``cache`` holds per-line index caches (a fresh
    dict is used when ``None``). Pass a prebuilt ``W`` to reuse a cached
    weight vector. Pass ``p_full`` (complete moving coords, input order) to
    also get the fully transformed structure for output; the reorder
    permutation is never applied to it, so output keeps input atom order.
    """
    if cache is None:
        cache = {}
    excl_arr = np.asarray(excl_arr, dtype=np.int64).ravel()
    P = np.asarray(P, dtype=np.float64)
    Pa = np.asarray(Pa)

    # center P at origin (Q arrives pre-centered)
    if nsatoms:
        pcenter = P[:natoms].mean(axis=0)
    else:
        pcenter = P.mean(axis=0)
    P = P - pcenter
    full = None
    if p_full is not None:
        full = np.asarray(p_full, dtype=np.float64) - pcenter

    if nsatoms:
        # solute-first superposition
        U = rmsd.kabsch(P[:natoms], Q[:natoms])
        P = P @ U
        if full is not None:
            full = full @ U

        if reorder is not None and not reorder_solvent_only:
            key = ("solute", len(P), natoms)
            entry = cache.get(key)
            if entry is None:
                soluexcl = excl_arr[excl_arr < natoms]
                soluteview = np.delete(np.arange(natoms), soluexcl)
                entry = (soluteview, soluexcl)
                cache[key] = entry
            else:
                soluteview, soluexcl = entry
            Pview = P[soluteview]
            Paview = Pa[soluteview]
            prr = _do_reorder(
                reorder,
                Qa[soluteview],
                Paview,
                Q[soluteview],
                Pview,
                cache,
                ("hq_solute", len(P), natoms),
            )
            Pview = Pview[prr]
            Paview = Paview[prr]
            Psolu = np.empty((natoms, 3), dtype=P.dtype)
            Psolu[soluteview] = Pview
            Psolu[soluexcl] = P[soluexcl]
            Pasolu = np.empty((natoms,), dtype=Pa.dtype)
            Pasolu[soluteview] = Paview
            Pasolu[soluexcl] = Pa[soluexcl]
            P = np.concatenate((Psolu, P[natoms:]))
            Pa = np.concatenate((Pasolu, Pa[natoms:]))
            U = rmsd.kabsch(P[:natoms], Q[:natoms])
            P = P @ U
            if full is not None:
                full = full @ U
    else:
        U = rmsd.kabsch(P, Q)
        P = P @ U
        if full is not None:
            full = full @ U

    if reorder is not None:
        key = ("solv", len(P), natoms)
        entry = cache.get(key)
        if entry is None:
            if nsatoms:
                exclusions = np.unique(np.concatenate((np.arange(natoms), excl_arr)))
            else:
                exclusions = np.unique(excl_arr)
            # keep exclusions in-bounds for varying sizes
            exclusions = exclusions[exclusions < len(P)]
            view = np.delete(np.arange(len(P)), exclusions)
            entry = (exclusions, view)
            cache[key] = entry
        else:
            exclusions, view = entry
        Pview = P[view]
        Paview = Pa[view]
        prr = _do_reorder(
            reorder, Qa[view], Paview, Q[view], Pview, cache, ("hq_solv", len(P), natoms)
        )
        Pview = Pview[prr]
        Paview = Paview[prr]
        Pr = scatter_reordered(len(P), view, exclusions, P[exclusions], Pview)
        Pa = scatter_reordered(len(Pa), view, exclusions, Pa[exclusions], Paview)
    else:
        Pr = P

    if weight_solute:
        if W is None:
            W = build_weight_vector(len(Pr), natoms, weight_solute)
    else:
        W = None

    if nsatoms and reorder is not None and not final_kabsch:
        return Aligned(Pr, Pa, W, None, None, "none", None, full)
    if weight_solute:
        R, _T, w_rmsd = rmsd.kabsch_weighted(Pr, Q, W)
        # NOTE on the upstream convention (verified numerically): R is the
        # Pr→Q rotation, but the returned translation belongs to the
        # transposed (Q→Pr) map, so ``Pr @ R + _T`` does NOT attain
        # ``w_rmsd``. Rebuild the exact Pr→Q translation from the weighted
        # centroids; ``Pr @ R + T`` then attains ``w_rmsd`` to ~1e-15.
        wsum = W.sum()
        cmp = (Pr * W[:, None]).sum(axis=0) / wsum
        cmq = (Q * W[:, None]).sum(axis=0) / wsum
        T = cmq - cmp @ R
        if full is not None:
            full = full @ R + T
        return Aligned(Pr, Pa, W, R, T, "weighted", float(w_rmsd), full)
    if full is not None:
        R = rmsd.kabsch(Pr, Q)
        full = full @ R
        return Aligned(Pr, Pa, W, R, np.zeros(3), "plain", None, full)
    return Aligned(Pr, Pa, W, None, None, "plain", None, full)
