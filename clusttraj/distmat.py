"""Functions to compute the RMSD matrix based on the provided
trajectory."""

from openbabel import pybel
import numpy as np
import rmsd
import os
import multiprocessing
import itertools
from .io import ClustOptions, Logger
from .utils import get_mol_info
from typing import List, Union, Callable


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


def build_distance_matrix(clust_opt: ClustOptions) -> np.ndarray:
    """Compute the RMSD matrix.

    Args:
        clust_opt (ClustOptions): The options for clustering.

    Returns:
        np.ndarray: The computed RMSD matrix.
    """
    # create iterator containing information to compute a line of the RMSD matrix
    inputiterator = zip(
        itertools.count(),
        map(
            lambda x: get_mol_info(x),
            pybel.readfile(
                os.path.splitext(clust_opt.trajfile)[1][1:], clust_opt.trajfile
            ),
        ),
        itertools.repeat(clust_opt.trajfile),
        itertools.repeat(clust_opt.no_hydrogen),
        itertools.repeat(clust_opt.reorder_alg),
        itertools.repeat(clust_opt.reorder_solvent_only),
        itertools.repeat(clust_opt.solute_natoms),
        itertools.repeat(clust_opt.weight_solute),
        itertools.repeat(clust_opt.reorder_excl),
        itertools.repeat(clust_opt.final_kabsch),
    )

    # create the pool with nprocs processes to compute the RMSD matrix in parallel
    p = multiprocessing.Pool(processes=clust_opt.n_workers)

    # build the RMSD matrix in parallel
    ldistmat = p.starmap(compute_distmat_line, inputiterator)

    return np.asarray([x for n in ldistmat if len(n) > 0 for x in n])


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
    # unpack q_info tuple
    q_atoms, q_all = q_info

    # get the number of non hydrogen atoms in the solute to subtract if needed
    natoms = nsatoms
    if noh:
        natoms = len(np.where(q_atoms[:nsatoms] != 1)[0])

    # initialize RMSD matrix
    distmat = []

    for idx2, mol2 in enumerate(
        pybel.readfile(os.path.splitext(trajfile)[1][1:], trajfile)
    ):
        # skip if it's not an element from the superior diagonal matrix
        if idx1 >= idx2:
            continue

        # arrays for second molecule
        p_atoms, p_all = get_mol_info(mol2)

        # consider the H or not consider depending on option
        if nsatoms:
            if noh:
                not_hydrogensP = np.where(p_atoms != 1)
                not_hydrogensQ = np.where(q_atoms != 1)
                P = p_all[not_hydrogensP]
                Q = q_all[not_hydrogensQ]
                Pa = p_atoms[not_hydrogensP]
                Qa = q_atoms[not_hydrogensQ]
            else:
                P = p_all
                Q = q_all
                Pa = p_atoms
                Qa = q_atoms

            pcenter = rmsd.centroid(P[:natoms])
            qcenter = rmsd.centroid(Q[:natoms])
        elif noh:
            not_hydrogensP = np.where(p_atoms != 1)
            not_hydrogensQ = np.where(q_atoms != 1)
            P = p_all[not_hydrogensP]
            Q = q_all[not_hydrogensQ]
            Pa = p_atoms[not_hydrogensP]
            Qa = q_atoms[not_hydrogensQ]
            pcenter = rmsd.centroid(P)
            qcenter = rmsd.centroid(Q)
        else:
            P = p_all
            Q = q_all
            Pa = p_atoms
            Qa = q_atoms
            pcenter = rmsd.centroid(P)
            qcenter = rmsd.centroid(Q)

        # center the coordinates at the origin
        P -= pcenter
        Q -= qcenter

        # generate rotation to superpose the solute configuration
        if nsatoms:
            # center the coordinates at the solute
            P -= rmsd.centroid(Q[:natoms])

            # try to improve atom matching by performing Kabsch
            # generate a rotation considering only the solute atoms without reordering
            U = rmsd.kabsch(P[:natoms], Q[:natoms])

            # rotate the whole system with this rotation
            P = np.dot(P, U)

            # reorder solute atoms
            if reorder and not reorder_solvent_only:
                # find the solute atoms that are not excluded
                soluexcl = np.where(reorderexcl < natoms)
                soluteview = np.delete(np.arange(natoms), reorderexcl[soluexcl])
                Pview = P[soluteview]
                Paview = Pa[soluteview]

                # reorder just these atoms
                prr = reorder(Qa[soluteview], Paview, Q[soluteview], Pview)
                Pview = Pview[prr]
                Paview = Paview[prr]

                # build the total structure reordering just these atoms
                # whereins = np.where(
                #     np.isin(np.arange(natoms), reorderexcl[soluexcl]) is True
                # )
                whereins = np.where(
                    np.atleast_1d(np.isin(np.arange(natoms), reorderexcl[soluexcl]))
                )
                Psolu = np.insert(
                    Pview,
                    [x - whereins[0].tolist().index(x) for x in whereins[0]],
                    P[reorderexcl[soluexcl]],
                    axis=0,
                )
                Pasolu = np.insert(
                    Paview,
                    [x - whereins[0].tolist().index(x) for x in whereins[0]],
                    Pa[reorderexcl[soluexcl]],
                    axis=0,
                )

                P = np.concatenate((Psolu, P[np.arange(len(P) - natoms) + natoms]))
                Pa = np.concatenate((Pasolu, Pa[np.arange(len(Pa) - natoms) + natoms]))

                # generate a rotation considering the reordered solute atoms
                U = rmsd.kabsch(P[:natoms], Q[:natoms])

                # rotate the whole system with this rotation
                P = np.dot(P, U)

        else:
            # Kabsch rotation
            U = rmsd.kabsch(P, Q)
            P = np.dot(P, U)

        # reorder the solvent atoms separately
        if reorder:
            # if the solute is specified, reorder just the solvent atoms in this step
            if nsatoms:
                exclusions = np.unique(np.concatenate((np.arange(natoms), reorderexcl)))
            else:
                exclusions = reorderexcl

            # get the view without the excluded atoms
            view = np.delete(np.arange(len(P)), exclusions)
            Pview = P[view]
            Paview = Pa[view]

            prr = reorder(Qa[view], Paview, Q[view], Pview)
            Pview = Pview[prr]

            # build the total molecule with the reordered atoms
            whereins = np.where(np.atleast_1d(np.isin(np.arange(len(P)), exclusions)))
            Pr = np.insert(
                Pview,
                [x - whereins[0].tolist().index(x) for x in whereins[0]],
                P[exclusions],
                axis=0,
            )

        else:
            Pr = P

        # compute the weights
        if weight_solute:
            W = np.zeros(Pr.shape[0])
            W[:natoms] = weight_solute / natoms
            W[natoms:] = (1.0 - weight_solute) / (Pr.shape[0] - natoms)

        # for solute solvent alignement, compute RMSD without Kabsch
        if nsatoms and reorder and not final_kabsch:
            if weight_solute:
                diff = Pr - Q
                distmat.append(np.sqrt(np.dot(W, np.sum(diff * diff, axis=1))))
            else:
                distmat.append(rmsd.rmsd(Pr, Q))
        else:
            if weight_solute:
                distmat.append(rmsd.kabsch_weighted_rmsd(Pr, Q, W))
            else:
                distmat.append(rmsd.kabsch_rmsd(Pr, Q))

    return distmat
