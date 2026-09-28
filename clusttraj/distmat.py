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
    align_core,
    build_weight_vector,
    filter_moving,
    prepare_ref,
    weighted_rmsd_no_kabsch,
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


def _fork_available() -> bool:
    """Whether child processes inherit memory (fork) instead of re-importing."""
    try:
        multiprocessing.get_context("fork")
        return True
    except (AttributeError, ValueError, RuntimeError):
        return False


def _new_pool(n_workers, **kwargs):
    """Pool preferring fork (fast startup, no re-import) with fallback."""
    if _fork_available():
        return multiprocessing.get_context("fork").Pool(processes=n_workers, **kwargs)
    return multiprocessing.Pool(processes=n_workers, **kwargs)


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
    excl = (
        np.asarray(clust_opt.reorder_excl, dtype=np.int64).ravel()
        if clust_opt.reorder_excl is not None
        else np.asarray([], dtype=np.int64)
    )

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
        if clust_opt.n_workers > 1 and not _fork_available():
            Logger.logger.warning(
                "Without fork (e.g. Windows/macOS) every worker process "
                "receives its own copy of the preloaded trajectory, so RAM "
                "use grows with the worker count; use preload=False "
                "(streaming) for large trajectories on these systems.\n"
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
                _compute_line_preloaded(
                    i, all_atoms[i], all_coords[i], all_atoms, all_coords, *worker_args
                )
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
            # submit in bounded batches: pool.starmap would otherwise turn
            # the len-less iterator into a list first, holding every frame's
            # arrays in the parent and defeating the streaming escape hatch
            batchsize = max(1, clust_opt.n_workers * 2)
            ldistmat = []
            with _new_pool(clust_opt.n_workers) as pool:
                batch = []
                for task in inputiterator:
                    batch.append(task)
                    if len(batch) >= batchsize:
                        ldistmat.extend(pool.starmap(compute_distmat_line, batch))
                        batch = []
                if batch:
                    ldistmat.extend(pool.starmap(compute_distmat_line, batch))

    return np.asarray([x for n in ldistmat if len(n) > 0 for x in n])


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
    """RMSD between one pair; Q must already be centered (hoisted per line).

    Every alignment step runs in :func:`_fastmath.align_core`, shared with
    ``io.align_mol``, so matrix values and saved structures always agree.
    Only the final scalar is computed here.
    """
    W = None
    if weight_solute:
        ckey = ("w", len(P), natoms, float(weight_solute))
        W = cache.get(ckey)
        if W is None:
            W = build_weight_vector(len(P), natoms, weight_solute)
            cache[ckey] = W
    al = align_core(
        P,
        Pa,
        Q,
        Qa,
        natoms,
        nsatoms=nsatoms,
        reorder=reorder,
        reorder_solvent_only=reorder_solvent_only,
        excl_arr=reorderexcl,
        weight_solute=weight_solute,
        final_kabsch=final_kabsch,
        cache=cache,
        W=W,
    )
    if al.kind == "none":
        if al.W is not None:
            return weighted_rmsd_no_kabsch(al.Pr, Q, al.W)
        return float(rmsd.rmsd(al.Pr, Q))
    if al.kind == "weighted":
        return float(al.w_rmsd)
    return float(rmsd.kabsch_rmsd(al.Pr, Q))


def _line_common(q_atoms, q_coords, noh, nsatoms, reorderexcl):
    """Shared per-line setup: filter + center Q, fresh cache, excl array."""
    Qa, Qref, natoms, _ = prepare_ref(q_atoms, q_coords, noh, nsatoms)
    excl_arr = (
        np.asarray(reorderexcl, dtype=np.int64).ravel()
        if reorderexcl is not None
        else np.asarray([], dtype=np.int64)
    )
    return Qa, Qref, natoms, excl_arr, {}


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
    Qa, Qref, natoms, excl_arr, cache = _line_common(
        q_atoms, q_coords, noh, nsatoms, reorderexcl
    )
    distmat: List[float] = []
    for idx2 in range(idx1 + 1, len(all_atoms)):
        Pa, P = filter_moving(all_atoms[idx2], all_coords[idx2], noh, nsatoms)
        Q = Qref.copy()  # cheap vs Hungarian; keeps kernel pure
        distmat.append(
            _pair_rmsd(
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
                cache,
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
    Qa, Qref, natoms, excl_arr, cache = _line_common(
        q_atoms, q_all, noh, nsatoms, reorderexcl
    )
    distmat: List[float] = []
    for idx2, mol2 in enumerate(pybel.readfile(_traj_format(trajfile), trajfile)):
        # skip if it's not an element from the superior diagonal matrix
        # (before building arrays: parsing alone is enough to advance)
        if idx1 >= idx2:
            continue
        p_atoms, p_all = get_mol_info(mol2)
        Pa, P = filter_moving(np.asarray(p_atoms), np.asarray(p_all), noh, nsatoms)
        Q = Qref.copy()
        distmat.append(
            _pair_rmsd(
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
                cache,
            )
        )
    return distmat
