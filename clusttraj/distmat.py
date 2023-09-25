"""Functions to compute the distance matrix based on the provided
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
    """Calculate or read a condensed distance matrix based on the given
    clustering options.

    Args:
        clust_opt (ClustOptions): The clustering options.

    Returns:
        np.ndarray: The condensed distance matrix.
    """
    # check if distance matrix will be read from input or calculated
    # if a file is specified, read it (TODO: check if the matrix makes sense)
    if clust_opt.input_distmat:
        Logger.logger.info(
            f"Reading condensed distance matrix from {clust_opt.distmat_name}\n"
        )
        distmat = np.load(clust_opt.distmat_name)
    # build a distance matrix already in the condensed form
    else:
        Logger.logger.info(
            f"Calculating distance matrix using {clust_opt.n_workers} threads\n"
        )
        distmat = build_distance_matrix(clust_opt)
        Logger.logger.info(
            f"Saving condensed distance matrix to {clust_opt.distmat_name}\n"
        )
        np.save(clust_opt.distmat_name, distmat)

    return distmat


def build_distance_matrix(clust_opt: ClustOptions) -> np.ndarray:
    """Compute the distance matrix.

    Args:
        clust_opt (ClustOptions): The options for clustering.

    Returns:
        np.ndarray: The computed distance matrix.
    """
    # create iterator containing information to compute a line of the distance matrix
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
        itertools.repeat(clust_opt.solute_natoms),
        itertools.repeat(clust_opt.reorder_excl),
        itertools.repeat(clust_opt.final_kabsch),
    )

    # create the pool with nprocs processes to compute the distance matrix in parallel
    p = multiprocessing.Pool(processes=clust_opt.n_workers)

    # build the distance matrix in parallel
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
    nsatoms: int,
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
        List[float]: The distance matrix.
    """  # noqa: E501
    # unpack q_info tuple
    q_atoms, q_all = q_info

    # get the number of non hydrogen atoms in the solute to subtract if needed
    natoms = nsatoms
    if noh:
        natoms = len(np.where(q_atoms[:nsatoms] != 1)[0])

    # initialize distance matrix
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
            if reorder:
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
                whereins = np.where(
                    np.isin(np.arange(natoms), reorderexcl[soluexcl]) is True
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

                # consider only the solvent atoms in the reorder (without exclusions)
                # solvexcl = np.where(reorderexcl >= natoms)
                # solvview = np.delete(np.arange(natoms, len(P)), reorderexcl[solvexcl])
                # Pview = P[solvview]
                # Paview = Pa[solvview]

                # # reorder just these atoms
                # prr = reorder(Qa[solvview], Paview, Q[solvview], Pview)
                # Pview = Pview[prr]
                # Paview = Paview[prr]

                # # build the total molecule with the reordered atoms
                # whereins = np.where(
                #     np.isin(np.arange(natoms, len(P)), reorderexcl[solvexcl]) == True
                # )
                # Psolv = np.insert(
                #     Pview,
                #     [x - whereins[0].tolist().index(x) for x in whereins[0]],
                #     P[reorderexcl[solvexcl]],
                #     axis=0,
                # )
                # Pasolv = np.insert(
                #     Paview,
                #     [x - whereins[0].tolist().index(x) for x in whereins[0]],
                #     Pa[reorderexcl[solvexcl]],
                #     axis=0,
                # )

                # Pr = np.concatenate((P[:natoms], Psolv))
                # Pra = np.concatenate((Pa[:natoms], Pasolv))
        else:
            # Kabsch rotation
            U = rmsd.kabsch(P, Q)
            P = np.dot(P, U)

        # reorder the solvent atoms separately
        if reorder:
            # get the view without the excluded atoms
            view = np.delete(np.arange(len(P)), reorderexcl)
            Pview = P[view]
            Paview = Pa[view]

            prr = reorder(Qa[view], Paview, Q[view], Pview)
            Pview = Pview[prr]

            # build the total molecule with the reordered atoms
            whereins = np.where(np.isin(np.arange(len(P)), reorderexcl) is True)
            Pr = np.insert(
                Pview,
                [x - whereins[0].tolist().index(x) for x in whereins[0]],
                P[reorderexcl],
                axis=0,
            )

        else:
            Pr = P

        # for solute solvent alignement, compute RMSD without Kabsch
        if nsatoms and reorder and not final_kabsch:
            distmat.append(rmsd.rmsd(Pr, Q))
        else:
            distmat.append(rmsd.kabsch_rmsd(Pr, Q))

    return distmat
