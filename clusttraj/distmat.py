from openbabel import pybel
from openbabel import openbabel
import numpy as np
import rmsd
import multiprocessing
import itertools
import logging


def get_distmat(clust_opt):
    # check if distance matrix will be read from input or calculated
    # if a file is specified, read it (TODO: check if the matrix makes sense)
    if clust_opt.input_distmat:
        logging.info(f'\nReading condensed distance matrix from {clust_opt.distmat_name}\n')
        distmat = np.load(clust_opt.distmat_name)
    # build a distance matrix already in the condensed form
    else:
        logging.info(f'\nCalculating distance matrix using {clust_opt.n_workers} threads\n')
        distmat = build_distance_matrix(clust_opt)
        logging.info(f'Saving condensed distance matrix to {clust_opt.distmat_name}\n')
        np.save(clust_opt.distmat_name, distmat)

    return distmat


def get_mol_coords(mol):
    q_all = []
    for atom in mol:
        q_all.append(atom.coords)

    return np.asarray(q_all)


def get_mol_info(mol):
    q_atoms = []
    q_all = []
    for atom in mol:
        q_atoms.append(openbabel.GetSymbol(atom.atomicnum))
        q_all.append(atom.coords)

    return np.asarray(q_atoms), np.asarray(q_all)


def build_distance_matrix(clust_opt):
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
    )

    # create the pool with nprocs processes to compute the distance matrix in parallel
    p = multiprocessing.Pool(processes=clust_opt.n_workers)

    # build the distance matrix in parallel
    ldistmat = p.starmap(compute_distmat_line, inputiterator)

    return np.asarray([x for n in ldistmat if len(n) > 0 for x in n])


def compute_distmat_line(idx1, q_info, trajfile, noh, reorder, nsatoms, reorderexcl):
    # unpack q_info tuple
    q_atoms, q_all = q_info

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
            # get the number of non hydrogen atoms in the solute to subtract if needed
            if noh:
                natoms = len(np.where(p_atoms[:nsatoms] != "H")[0])
            else:
                natoms = nsatoms

            if noh:
                not_hydrogensP = np.where(p_atoms != "H")
                not_hydrogensQ = np.where(q_atoms != "H")
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
            not_hydrogensP = np.where(p_atoms != "H")
            not_hydrogensQ = np.where(q_atoms != "H")
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
            # generate a rotation considering only the solute atoms
            U = rmsd.kabsch(P[:natoms], Q[:natoms])

            # rotate the whole system with this rotation
            P = np.dot(P, U)

            # reorder solute atoms
            # find the solute atoms that are not excluded
            soluexcl = np.where(reorderexcl < natoms)
            soluteview = np.delete(np.arange(natoms), reorderexcl[soluexcl])
            Pview = P[soluteview]
            Paview = Pa[soluteview]

            # reorder just these atoms
            prr = reorder(Qa[soluteview], Paview, Q[soluteview], Pview)
            Pview = Pview[prr]
            Paview = Paview[prr]

            # build the total molecule reordering just these atoms
            whereins = np.where(
                np.isin(np.arange(natoms), reorderexcl[soluexcl]) == True
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

            # generate a rotation considering only the solute atoms
            U = rmsd.kabsch(P[:natoms], Q[:natoms])

            # rotate the whole system with this rotation
            P = np.dot(P, U)

            # consider only the solvent atoms in the reorder (without exclusions)
            solvexcl = np.where(reorderexcl >= natoms)
            solvview = np.delete(np.arange(natoms, len(P)), reorderexcl[solvexcl])
            Pview = P[solvview]
            Paview = Pa[solvview]

            # reorder just these atoms
            prr = reorder(Qa[solvview], Paview, Q[solvview], Pview)
            Pview = Pview[prr]
            Paview = Paview[prr]

            # build the total molecule with the reordered atoms
            whereins = np.where(
                np.isin(np.arange(natoms, len(P)), reorderexcl[solvexcl]) == True
            )
            Psolv = np.insert(
                Pview,
                [x - whereins[0].tolist().index(x) for x in whereins[0]],
                P[reorderexcl[solvexcl]],
                axis=0,
            )
            Pasolv = np.insert(
                Paview,
                [x - whereins[0].tolist().index(x) for x in whereins[0]],
                Pa[reorderexcl[solvexcl]],
                axis=0,
            )

            Pr = np.concatenate((P[:natoms], Psolv))
            Pra = np.concatenate((Pa[:natoms], Pasolv))

        # reorder the atoms if necessary
        elif reorder:
            # try to improve atom matching by performing Kabsch
            # generate a rotation considering only the solute atoms
            U = rmsd.kabsch(P, Q)

            # rotate the whole system with this rotation
            P = np.dot(P, U)

            # get the view without the excluded atoms
            view = np.delete(np.arange(len(P)), reorderexcl)
            Pview = P[view]
            Paview = Pa[view]

            prr = reorder(Qa[view], Paview, Q[view], Pview)
            Pview = Pview[prr]
            Paview = Paview[prr]

            # build the total molecule with the reordered atoms
            whereins = np.where(np.isin(np.arange(len(P)), reorderexcl) == True)
            Pr = np.insert(
                Pview,
                [x - whereins[0].tolist().index(x) for x in whereins[0]],
                P[reorderexcl],
                axis=0,
            )
            Pra = np.insert(
                Paview,
                [x - whereins[0].tolist().index(x) for x in whereins[0]],
                Pa[reorderexcl],
                axis=0,
            )

        else:
            Pr = P
            Pra = Pa

        # get the RMSD and store it
        distmat.append(rmsd.kabsch_rmsd(Pr, Q))

    return distmat
