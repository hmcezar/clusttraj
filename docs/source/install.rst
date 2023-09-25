Installing ClustTraj
====================

Installing ``clusttraj`` is simple and can be achieved using ``pip``:

.. code-block:: console

    pip install clusttraj

This will automatically install the package and its dependencies.

We recommend performing the installation in an empty virtual environment.

Dependencies
************

``clustttraj`` dependencies should be installed automatically by ``pip``.

Currently, the following dependencies are installed:

* `NumPy <http://www.numpy.org/>`_
* `OpenBabel <http://openbabel.org/>`_
* `RMSD <https://github.com/charnley/rmsd>`_
* `QML <https://github.com/qmlcode/qml>`_
* `SciPy <https://www.scipy.org/>`_
* `scikit-learn <http://scikit-learn.org/stable/index.html>`_
* `matplotlib <https://matplotlib.org/>`_

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