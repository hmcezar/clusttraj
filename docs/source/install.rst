Installing ClustTraj
====================

Installing ``clusttraj`` is simple and can be achieved using ``pip``:

.. code-block:: console

    pip install clusttraj

This will automatically install ``clusttraj`` and its Python dependencies, except for
Open Babel.

We recommend performing the installation in an empty virtual environment.

Dependencies
************

Most ``clusttraj`` dependencies are installed automatically by ``pip``.
Open Babel is a runtime dependency, but it is not installed by default.

Currently, the following dependencies are used:

* `NumPy <http://www.numpy.org/>`_
* `OpenBabel <http://openbabel.org/>`_
* `RMSD <https://github.com/charnley/rmsd>`_
* `SciPy <https://www.scipy.org/>`_
* `scikit-learn <http://scikit-learn.org/stable/index.html>`_
* `matplotlib <https://matplotlib.org/>`_

If you use Conda, install Open Babel from conda-forge before installing
``clusttraj``:

.. code-block:: console

    conda install -c conda-forge openbabel
    pip install clusttraj

For pip-only environments, ``clusttraj`` provides an optional dependency that
installs the ``openbabel-wheel`` package:

.. code-block:: console

    pip install "clusttraj[openbabel]"

Avoid mixing Conda Open Babel and ``openbabel-wheel`` in the same environment.
If you see Open Babel import or linker errors, remove one provider and reinstall
Open Babel from the package manager used by that environment.

``qmllib`` is an optional dependency for one of the reordering algorithms and
can be installed with:

.. code-block:: console

    pip install "clusttraj[qml]"

Since the ``qml`` project development has been slow, we provide a fork repository in which
we updated the package to be installable in modern enviroments with newer versions of 
Python and libraries.
This modified version can be downloaded and installed from `this link <https://github.com/hmcezar/qml>`_.

Installation Problems
*********************

If you have problems installing ``clusttraj`` because installing ``qml`` fails, try installing 
``qml`` yourself first.
For Python 3.11, you might have to either disable setuptools distutils setting the environment 
variable ``SETUPTOOLS_USE_DISTUTILS=stdlib`` before installing, or downgrading ``setuptools``
to a version prior than 60.0.
For example, you could install ``qml`` with:

.. code-block:: console
    
    pip install "setuptools<60"
    pip install "qml @ git+https://github.com/hmcezar/qml@develop"
