"""Saved structures must carry exactly the RMSD-matrix superposition.

``distmat`` (matrix values) and ``io.align_mol`` (structures written to
disk) share one kernel (``_fastmath.align_core``). These tests pin the
contract from three sides, over an option grid and every pair of
``testtraj.xyz``:

1. Without reorder, the plain metric between a saved structure and the
   prepared reference reproduces the matrix entry end to end.
2. For every combo (including reorder/weights), the coordinates written
   by ``align_mol`` equal the kernel output bit for bit.
3. For every combo, the kernel outputs reconstruct the matrix entry with
   plain operations (no fitting), proving the written transform *is* the
   matrix superposition.

Any drift between the matrix path and the save path fails these tests.
"""

import copy

import numpy as np
import pytest
import rmsd
from openbabel import openbabel, pybel
from scipy.spatial.distance import squareform

from clusttraj._fastmath import (
    align_core,
    filter_moving,
    prepare_ref,
    weighted_rmsd_no_kabsch,
)
from clusttraj.distmat import build_distance_matrix
from clusttraj.io import ClustOptions, align_mol
from clusttraj.utils import get_mol_info

TRAJ = "test/ref/testtraj.xyz"
HUNG = rmsd.reorder_hungarian

COMBOS = [
    dict(noh=True, reorder=None, solv=False, ns=17, ws=None, excl=[], fk=False),
    dict(noh=False, reorder=None, solv=False, ns=17, ws=None, excl=[], fk=False),
    dict(noh=True, reorder=None, solv=False, ns=17, ws=None, excl=[], fk=True),
    dict(noh=False, reorder=HUNG, solv=False, ns=17, ws=None, excl=[], fk=False),
    dict(noh=True, reorder=HUNG, solv=False, ns=17, ws=None, excl=[], fk=True),
    dict(noh=False, reorder=HUNG, solv=True, ns=17, ws=0.8, excl=[1, 2, 3], fk=True),
    dict(noh=False, reorder=HUNG, solv=True, ns=17, ws=0.9, excl=[], fk=False),
    dict(noh=True, reorder=HUNG, solv=True, ns=17, ws=None, excl=[], fk=True),
    dict(noh=False, reorder=HUNG, solv=True, ns=17, ws=0.5, excl=[0], fk=True),
]

NO_REORDER = [c for c in COMBOS if c["reorder"] is None]


def parse_xyz(molstring):
    lines = molstring.strip().split("\n")
    nat = int(lines[0])
    symbols = [row.split()[0] for row in lines[2 : 2 + nat]]
    coords = np.array([[float(x) for x in row.split()[1:]] for row in lines[2 : 2 + nat]])
    return lines[1], symbols, coords


def ref_of(raw, combo):
    return prepare_ref(raw[0], raw[1], combo["noh"], combo["ns"])


def square_matrix(options_dict, combo):
    opt = copy.deepcopy(options_dict)
    opt.update(
        no_hydrogen=combo["noh"],
        reorder_alg=combo["reorder"],
        reorder_solvent_only=combo["solv"],
        solute_natoms=combo["ns"],
        weight_solute=combo["ws"],
        reorder_excl=combo["excl"],
        final_kabsch=combo["fk"],
    )
    return squareform(build_distance_matrix(ClustOptions(**opt)))


def filter_full(coords, atoms, noh):
    atoms = np.asarray(atoms)
    if noh:
        return coords[atoms != 1]
    return coords


def test_saved_structures_reproduce_matrix(options_dict):
    """End to end (no reorder): saved frame vs reference gives the entry.

    Uses the rotation-free ``rmsd.rmsd``, so this only passes if the saved
    structure already carries the optimal superposition.
    """
    mols = list(pybel.readfile("xyz", TRAJ))
    raws = [get_mol_info(mol) for mol in mols]
    n = len(mols)
    assert n == 3

    for combo in NO_REORDER:
        sq = square_matrix(options_dict, combo)
        for i in range(n):
            i_atoms = np.asarray(raws[i][0])
            Qa_i, Qref_i, _, _ = ref_of(raws[i], combo)
            for j in range(i + 1, n):
                molstring = align_mol(
                    mols[j],
                    Qref_i,
                    Qa_i,
                    combo["noh"],
                    combo["reorder"],
                    combo["solv"],
                    combo["ns"],
                    combo["ws"],
                    combo["excl"],
                    combo["fk"],
                )
                title, symbols, coords = parse_xyz(molstring)

                # output keeps input order, atoms and title intact
                j_atoms = np.asarray(raws[j][0])
                assert title == mols[j].title.rstrip()
                assert symbols == [openbabel.GetSymbol(int(a)) for a in j_atoms]
                assert len(coords) == len(j_atoms)

                coords = filter_full(coords, j_atoms, combo["noh"])
                assert len(coords) == len(Qref_i)
                assert float(rmsd.rmsd(coords, Qref_i)) == pytest.approx(
                    sq[i, j], abs=1e-8
                )


def test_written_coords_equal_kernel(options_dict):
    """Written coordinates are the kernel output (all combos)."""
    mols = list(pybel.readfile("xyz", TRAJ))
    raws = [get_mol_info(mol) for mol in mols]
    n = len(mols)

    for combo in COMBOS:
        for i in range(n):
            Qa_i, Qref_i, natoms_i, _ = ref_of(raws[i], combo)
            for j in range(i + 1, n):
                molstring = align_mol(
                    mols[j],
                    Qref_i,
                    Qa_i,
                    combo["noh"],
                    combo["reorder"],
                    combo["solv"],
                    combo["ns"],
                    combo["ws"],
                    combo["excl"],
                    combo["fk"],
                )
                _, _, coords = parse_xyz(molstring)

                p_atoms = np.asarray(raws[j][0])
                p_all = np.asarray(raws[j][1], dtype=np.float64)
                Pa, P = filter_moving(p_atoms, p_all, combo["noh"], combo["ns"])
                excl = np.asarray(combo["excl"], dtype=np.int64)
                al = align_core(
                    P,
                    Pa,
                    Qref_i,
                    Qa_i,
                    natoms_i,
                    nsatoms=combo["ns"],
                    reorder=combo["reorder"],
                    reorder_solvent_only=combo["solv"],
                    excl_arr=excl,
                    weight_solute=combo["ws"],
                    final_kabsch=combo["fk"],
                    p_full=p_all,
                )
                assert coords == pytest.approx(al.full, abs=1e-9)


def test_kernel_reproduces_matrix(options_dict):
    """Kernel outputs rebuild the matrix entry with plain ops (all combos)."""
    mols = list(pybel.readfile("xyz", TRAJ))
    raws = [get_mol_info(mol) for mol in mols]
    n = len(mols)

    for combo in COMBOS:
        sq = square_matrix(options_dict, combo)
        for i in range(n):
            Qa_i, Qref_i, natoms_i, _ = ref_of(raws[i], combo)
            for j in range(i + 1, n):
                p_atoms = np.asarray(raws[j][0])
                p_all = np.asarray(raws[j][1], dtype=np.float64)
                Pa, P = filter_moving(p_atoms, p_all, combo["noh"], combo["ns"])
                excl = np.asarray(combo["excl"], dtype=np.int64)
                al = align_core(
                    P,
                    Pa,
                    Qref_i,
                    Qa_i,
                    natoms_i,
                    nsatoms=combo["ns"],
                    reorder=combo["reorder"],
                    reorder_solvent_only=combo["solv"],
                    excl_arr=excl,
                    weight_solute=combo["ws"],
                    final_kabsch=combo["fk"],
                    p_full=p_all,
                )
                if al.kind == "none":
                    if al.W is not None:
                        value = weighted_rmsd_no_kabsch(al.Pr, Qref_i, al.W)
                    else:
                        value = float(rmsd.rmsd(al.Pr, Qref_i))
                elif al.kind == "weighted":
                    value = weighted_rmsd_no_kabsch(
                        al.Pr @ al.R + al.T, Qref_i, al.W
                    )
                else:
                    value = float(rmsd.rmsd(al.Pr @ al.R, Qref_i))
                assert value == pytest.approx(sq[i, j], abs=1e-9)
