Introduction
============

``clusttraj`` is a Python package which aims to cluster similar configurations of molecular dynamics or Monte Carlo simulation trajectories.

The optimal alignment between each snapshot is sought using reordering algorithms and optimal rotations.

Any format supported by `openbabel <https://github.com/openbabel/openbabel>`_ is accepted as input or output, including the popular .xyz, .pdb and .xtc.

Motivation
**********

The idea of ``clusttraj`` is to be an easy to use, and easy to extend platform for clustering trajectories.

We also focus in providing options for the analysis of solute-solvent systems.

Limitations
***********

``clusttraj`` is limited by the agglomerative clustering algorithms provided by `SciPy <https://www.scipy.org/>`_.

However, there are many algorithms implemented for one to choose, as can be seen `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_.
