"""Functions to compute the RMSD matrix based on the provided
trajectory."""

import numpy as np
import rmsd
import os
import itertools
import multiprocessing
from .io import ClustOptions, Logger
from openbabel import pybel
from .utils import get_mol_info
from ._fastmath import (
    hungarian_ref_groups,
    kabsch_weighted_rmsd_fast,
    reorder_hungarian_refcached,
    weighted_rmsd_no_kabsch,
    build_weight_vector,
    scatter_reordered,
)
from typing import List, Union, Callable, Tuple

#: Warn (but still proceed) when preloading a trajectory file larger than this.
PRELOAD_WARN_BYTES = 1_000_000_000


def get_distmat(clust_opt: ClustOptions) -> np.ndarray:
    """Calculate or read a condensed RMSD matrix based on the given
    clustering options.

    Args:
        clust_opt (ClustOptions): The clustering options.

    Returns:
        np.ndarray: The condensed RMSD matrix.
    """
    # check if RMSD matrix will be read from input or calculated
    # if a file is specified, read it (TODO: check if the matrix makes sense)
    if clust_opt.input_distmat:
        Logger.logger.info(
            f"Reading condensed RMSD matrix from {clust_opt.distmat_name}\n"
        )
        distmat = np.load(clust_opt.distmat_name)
    # build a RMSD matrix already in the condensed form
    else:
        Logger.logger.info(
            f"Calculating RMSD matrix using {clust_opt.n_workers} threads\n"
        )
        distmat = build_distance_matrix(clust_opt)
        Logger.logger.info(f"Saving condensed RMSD matrix to {clust_opt.distmat_name}\n")
        np.save(clust_opt.distmat_name, distmat)

    return distmat


def _traj_format(trajfile: str) -> str:
    return os.path.splitext(trajfile)[1][1:]


def load_trajectory_arrays(trajfile: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Read the whole trajectory into memory.

    Returns two parallel lists: atomic numbers and coordinates per frame.
    Only used when ``preload=True`` (the default): this parses O(N) once
    instead of re-parsing the file per matrix line, but holds all frames in
    RAM — pass ``--no-preload`` for trajectories too large to fit in memory.
    """
    fmt = _traj_format(trajfile)
    all_atoms: List[np.ndarray] = []
    all_coords: List[np.ndarray] = []
    for mol in pybel.readfile(fmt, trajfile):
        a, c = get_mol_info(mol)
        all_atoms.append(np.asarray(a))
        all_coords.append(np.asarray(c, dtype=np.float64))
    return all_atoms, all_coords


def _new_pool(n_workers, **kwargs):
    """Pool preferring fork (fast startup, no re-import) with fallback."""
    try:
        ctx = multiprocessing.get_context("fork")
    except (AttributeError, ValueError, RuntimeError):
        return multiprocessing.Pool(processes=n_workers, **kwargs)
    return ctx.Pool(processes=n_workers, **kwargs)


# --- worker state for the preloaded pool path ---
_WORK_ATOMS = None
_WORK_COORDS = None
_WORK_NOH = False
_WORK_REORDER = None
_WORK_SOLV_ONLY = False
_WORK_NSATOMS = 0
_WORK_WS = None
_WORK_EXCL = None
_WORK_FKABSCH = False


def _init_worker(atoms, coords, noh, reorder, solv_only, nsatoms, ws, excl, fkabsch):
    global _WORK_ATOMS, _WORK_COORDS, _WORK_NOH, _WORK_REORDER
    global _WORK_SOLV_ONLY, _WORK_NSATOMS, _WORK_WS, _WORK_EXCL, _WORK_FKABSCH
    _WORK_ATOMS = atoms
    _WORK_COORDS = coords
    _WORK_NOH = noh
    _WORK_REORDER = reorder
    _WORK_SOLV_ONLY = solv_only
    _WORK_NSATOMS = nsatoms
    _WORK_WS = ws
    _WORK_EXCL = excl
    _WORK_FKABSCH = fkabsch


def _worker_line(idx1: int) -> List[float]:
    return _compute_line_preloaded(
        idx1,
        _WORK_ATOMS[idx1],
        _WORK_COORDS[idx1],
        _WORK_ATOMS,
        _WORK_COORDS,
        _WORK_NOH,
        _WORK_REORDER,
        _WORK_SOLV_ONLY,
        _WORK_NSATOMS,
        _WORK_WS,
        _WORK_EXCL,
        _WORK_FKABSCH,
    )


def build_distance_matrix(clust_opt: ClustOptions) -> np.ndarray:
    """Compute the RMSD matrix.

    Two paths, selected by ``clust_opt.preload`` (default ``True``):

    * ``preload=True`` (default): the whole trajectory is parsed once into
      RAM and reused for every pair (O(N) parses instead of O(N^2)). Much
      faster for trajectories that fit in memory (e.g. the MOx methanol
      case: ~6x). A warning is logged for files over ~1 GB.
    * ``preload=False`` (opt-out via ``--no-preload``): the trajectory is
      streamed from disk; each worker only holds two frames at a time.
      Suitable for trajectories of any size, including dozens of GB.

    Args:
        clust_opt (ClustOptions): The options for clustering.

    Returns:
        np.ndarray: The computed RMSD matrix.
    """
    excl = np.asarray(clust_opt.reorder_excl, dtype=np.int64).ravel()

    if bool(getattr(clust_opt, "preload", True)):
        try:
            fsize = os.path.getsize(clust_opt.trajfile)
        except OSError:
            fsize = 0
        if fsize > PRELOAD_WARN_BYTES:
            Logger.logger.info(
                f"Preloading trajectory ({fsize / 1e9:.1f} GB) into memory; "
                "use preload=False (streaming) if RAM is limited.\n"
            )
        all_atoms, all_coords = load_trajectory_arrays(clust_opt.trajfile)
        n_frames = len(all_atoms)
        worker_args = (
            clust_opt.no_hydrogen,
            clust_opt.reorder_alg,
            clust_opt.reorder_solvent_only,
            clust_opt.solute_natoms,
            clust_opt.weight_solute,
            excl,
            clust_opt.final_kabsch,
        )
        if clust_opt.n_workers == 1:
            ldistmat = [
                _compute_line_preloaded(i, all_atoms[i], all_coords[i], all_atoms, all_coords, *worker_args)
                for i in range(n_frames)
            ]
        else:
            with _new_pool(
                clust_opt.n_workers,
                initializer=_init_worker,
                initargs=(all_atoms, all_coords, *worker_args),
            ) as pool:
                ldistmat = pool.map(_worker_line, range(n_frames))
    else:
        # streaming path: one task per matrix line, file re-read per line
        # (memory use is O(1) frames; slower than preload for small trajs)
        fmt = _traj_format(clust_opt.trajfile)
        inputiterator = zip(
            itertools.count(),
            map(get_mol_info, pybel.readfile(fmt, clust_opt.trajfile)),
            itertools.repeat(clust_opt.trajfile),
            itertools.repeat(clust_opt.no_hydrogen),
            itertools.repeat(clust_opt.reorder_alg),
            itertools.repeat(clust_opt.reorder_solvent_only),
            itertools.repeat(clust_opt.solute_natoms),
            itertools.repeat(clust_opt.weight_solute),
            itertools.repeat(excl),
            itertools.repeat(clust_opt.final_kabsch),
        )
        if clust_opt.n_workers == 1:
            ldistmat = [compute_distmat_line(*task) for task in inputiterator]
        else:
            with _new_pool(processes=clust_opt.n_workers) as pool:
                ldistmat = pool.starmap(compute_distmat_line, inputiterator)

    return np.asarray([x for n in ldistmat if len(n) > 0 for x in n])


def _prepare_q_side(q_atoms, q_all, noh, nsatoms):
    """Filter hydrogens once per line; return Qa, Q, natoms."""
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
    return Qa, Q, natoms


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


def _pair_rmsd(
    P,
    Pa,
    Q,
    Qa,
    natoms,
    nsatoms,
    reorder,
    reorder_solvent_only,
    reorderexcl,
    weight_solute,
    final_kabsch,
    cache,
) -> float:
    """RMSD between one pair; Q must already be centered (hoisted per line)."""
    # center P at origin (Q arrives pre-centered from the line cache)
    if nsatoms:
        pcenter = P[:natoms].mean(axis=0)
    else:
        pcenter = P.mean(axis=0)
    P = P - pcenter

    if nsatoms:
        # solute-first superposition (Q solute already at origin, so the old
        # residual centering here was a ~1e-16 no-op; skipped)
        U = rmsd.kabsch(P[:natoms], Q[:natoms])
        P = P @ U

        if reorder is not None and not reorder_solvent_only:
            key = ("solute", len(P), natoms)
            entry = cache.get(key)
            if entry is None:
                excl_arr = np.asarray(reorderexcl, dtype=np.int64).ravel()
                soluexcl = excl_arr[excl_arr < natoms]
                soluteview = np.delete(np.arange(natoms), soluexcl)
                entry = (soluteview, soluexcl)
                cache[key] = entry
            else:
                soluteview, soluexcl = entry
            Pview = P[soluteview]
            Paview = Pa[soluteview]
            prr = _do_reorder(reorder, Qa[soluteview], Paview, Q[soluteview], Pview, cache, ("hq_solute", len(P), natoms))
            Pview = Pview[prr]
            Paview = Paview[prr]
            # scatter back (exact replacement for insert+index loop)
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
    else:
        U = rmsd.kabsch(P, Q)
        P = P @ U

    if reorder is not None:
        key = ("solv", len(P), natoms)
        entry = cache.get(key)
        if entry is None:
            excl_arr = np.asarray(reorderexcl, dtype=np.int64).ravel()
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
        prr = _do_reorder(reorder, Qa[view], Paview, Q[view], Pview, cache, ("hq_solv", len(P), natoms))
        Pview = Pview[prr]
        Pr = scatter_reordered(len(P), view, exclusions, P[exclusions], Pview)
    else:
        Pr = P

    if weight_solute:
        ckey = ("w", len(Pr), natoms, float(weight_solute))
        W = cache.get(ckey)
        if W is None:
            W = build_weight_vector(len(Pr), natoms, weight_solute)
            cache[ckey] = W

    if nsatoms and reorder is not None and not final_kabsch:
        if weight_solute:
            return weighted_rmsd_no_kabsch(Pr, Q, W)
        return float(rmsd.rmsd(Pr, Q))
    if weight_solute:
        return float(kabsch_weighted_rmsd_fast(Pr, Q, W))
    return float(rmsd.kabsch_rmsd(Pr, Q))


def _line_common(q_atoms, q_coords, noh, nsatoms, reorderexcl):
    """Shared per-line setup: filter + center Q, fresh cache, excl array."""
    q_atoms = np.asarray(q_atoms)
    q_coords = np.asarray(q_coords, dtype=np.float64)
    Qa, Qref, natoms = _prepare_q_side(q_atoms, q_coords, noh, nsatoms)
    if nsatoms:
        qcenter = Qref[:natoms].mean(axis=0) if len(Qref) else 0.0
    else:
        qcenter = Qref.mean(axis=0) if len(Qref) else 0.0
    Qref = Qref - qcenter
    excl_arr = np.asarray(reorderexcl, dtype=np.int64).ravel() if reorderexcl is not None else np.asarray([], dtype=np.int64)
    return Qa, Qref, natoms, excl_arr, {}


def _filter_p_side(p_atoms, p_all, noh, nsatoms):
    """Filter hydrogens for the moving frame; returns Pa, P (fresh copy)."""
    if nsatoms:
        if noh:
            p_mask = p_atoms != 1
            return p_atoms[p_mask], p_all[p_mask].astype(np.float64, copy=True)
        return p_atoms, np.array(p_all, dtype=np.float64, copy=True)
    if noh:
        p_mask = p_atoms != 1
        return p_atoms[p_mask], p_all[p_mask].astype(np.float64, copy=True)
    return p_atoms, np.array(p_all, dtype=np.float64, copy=True)


def _compute_line_preloaded(
    idx1,
    q_atoms,
    q_coords,
    all_atoms,
    all_coords,
    noh,
    reorder,
    reorder_solvent_only,
    nsatoms,
    weight_solute,
    reorderexcl,
    final_kabsch,
) -> List[float]:
    """One matrix line from preloaded arrays (preload=True path)."""
    Qa, Qref, natoms, excl_arr, cache = _line_common(q_atoms, q_coords, noh, nsatoms, reorderexcl)
    distmat: List[float] = []
    for idx2 in range(idx1 + 1, len(all_atoms)):
        Pa, P = _filter_p_side(all_atoms[idx2], all_coords[idx2], noh, nsatoms)
        Q = Qref.copy()  # cheap vs Hungarian; keeps kernel pure
        distmat.append(
            _pair_rmsd(
                P, Pa, Q, Qa, natoms, nsatoms, reorder,
                reorder_solvent_only, excl_arr, weight_solute,
                final_kabsch, cache,
            )
        )
    return distmat


def compute_distmat_line(
    idx1: int,
    q_info: tuple,
    trajfile: str,
    noh: bool,
    reorder: Union[
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray], None
    ],
    reorder_solvent_only: bool,
    nsatoms: int,
    weight_solute: float,
    reorderexcl: np.ndarray,
    final_kabsch: bool,
) -> List[float]:
    """Compute the distance between molecule idx1 and molecules with idx2 >
    idx1.

    Streaming implementation: the trajectory file is read once per call and
    only two frames are ever held in memory, so arbitrarily large
    trajectories can be processed. Frames with ``idx2 <= idx1`` are parsed
    but skipped before building their atom/coordinate arrays.

    Args:
        idx1 (int): The index of the first molecule.
        q_info (tuple): Tuple containing the atom and all information of the first molecule.
        trajfile (str): The path to the trajectory file.
        noh (bool): Whether to consider hydrogen atoms or not.
        reorder (Union[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray], None]):
            A function to reorder the atoms, if necessary.
        nsatoms (int): The number of atoms in the solute.
        reorderexcl (np.ndarray): The array defining the excluded atoms during reordering.
        final_kabsch (bool): Whether to perform the final Kabsch rotation or not.

    Returns:
        List[float]: The RMSD matrix.
    """  # noqa: E501
    q_atoms, q_all = q_info
    Qa, Qref, natoms, excl_arr, cache = _line_common(q_atoms, q_all, noh, nsatoms, reorderexcl)
    distmat: List[float] = []
    for idx2, mol2 in enumerate(pybel.readfile(_traj_format(trajfile), trajfile)):
        # skip if it's not an element from the superior diagonal matrix
        # (before building arrays: parsing alone is enough to advance)
        if idx1 >= idx2:
            continue
        p_atoms, p_all = get_mol_info(mol2)
        Pa, P = _filter_p_side(np.asarray(p_atoms), np.asarray(p_all), noh, nsatoms)
        Q = Qref.copy()
        distmat.append(
            _pair_rmsd(
                P, Pa, Q, Qa, natoms, nsatoms, reorder,
                reorder_solvent_only, excl_arr, weight_solute,
                final_kabsch, cache,
            )
        )
    return distmat
