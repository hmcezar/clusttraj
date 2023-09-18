import pytest
import numpy as np
from clusttraj.utils import get_mol_coords, get_mol_info


def test_get_mol_coords(pybel_mol):
    coords = get_mol_coords(pybel_mol)

    expected_coords = np.array(
        [[0.0, 0.0, 0.11779], [0.0, 0.75545, -0.47116], [0.0, -0.75545, -0.47116]]
    )
    assert coords == pytest.approx(expected_coords, abs=1e-8)


def test_get_mol_info(pybel_mol):
    atomicnums, coords = get_mol_info(pybel_mol)

    expected_atomicnums = np.array([8, 1, 1])
    expected_coords = np.array(
        [[0.0, 0.0, 0.11779], [0.0, 0.75545, -0.47116], [0.0, -0.75545, -0.47116]]
    )

    assert atomicnums == pytest.approx(expected_atomicnums, abs=1e-8)
    assert coords == pytest.approx(expected_coords, abs=1e-8)
