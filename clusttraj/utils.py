from openbabel import openbabel
import numpy as np

def get_mol_coords(mol):
    q_all = []
    for atom in mol:
        q_all.append(atom.coords)

    return np.asarray(q_all)


def get_mol_info(mol):
    q_atoms = []
    q_all = []
    for atom in mol:
        q_atoms.append(atom.atomicnum)
        q_all.append(atom.coords)

    return np.array(q_atoms), np.array(q_all)