"""Additional utility functions."""

from openbabel import pybel
import numpy as np
from typing import Tuple


def get_mol_coords(mol: pybel.Molecule) -> np.ndarray:
    """Get the coordinates of all atoms in a molecule.

    Args:
        mol (pybel.Molecule): The molecule object.

    Returns:
        np.ndarray: The array of atom coordinates.
    """
    return np.asarray([atom.coords for atom in mol])


def get_mol_info(mol: pybel.Molecule) -> Tuple[np.ndarray, np.ndarray]:
    """Get the atomic numbers and coordinates of all atoms in a molecule.

    Args:
        mol (pybel.Molecule): The molecule object.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The array of atomic numbers and the array of atom coordinates.
    """
    return np.array([atom.atomicnum for atom in mol]), np.array(
        [atom.coords for atom in mol]
    )
