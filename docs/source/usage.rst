Usage
=====

To see all the options run the script with the ``-h`` command option:

.. code-block:: console
	
	clusttraj -h


or

.. code-block:: console

	python -m clusttraj -h

The only mandatory argument are the path to the file containing the trajectory (in a format that OpenBabel can read with Pybel) and the maximum RMSD between two configurations for them to be considered of the same cluster.

.. code-block:: console

	clusttraj trajectory.xyz -rmsd <threshold>

Instead of fixing the RMSD, one can use the ``-ss`` flag to determine the threshold as the value that maximize the `silhouette coefficient <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_ metric:

.. code-block:: console

	clusttraj trajectory.xyz -ss

Additional options are available for specifying the input and output files and selecting how the clustering is done. The possible methods used for the agglomerative clustering are the ones available in the linkage method of SciPy's hierarchical clustering. A list with the possible methods (selected with ``-m``) and the description of each of them can be found `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_.

The default method for the linkage is ``average``, since `it was found <https://dx.doi.org/10.1021/ct700119m>`_ to have a good compromise with the number of clusters and the actual similarity. To learn more about how the clustering is performed using this algorithm, see `UPGMA <https://en.wikipedia.org/wiki/UPGMA>`_.

If the ``-n`` option is used, the hydrogens are ignored when performing the Kabsch algorithm to find the superposition and calculating the RMSD. This is useful to avoid clustering identical structures with just a methyl group rotated as different.

The ``-e`` or ``--reorder`` option, tries to reorder the atoms to increase the overlap and reduce the RMSD. The algorithm can be selected with ``--reorder-alg``, between qml (default), hungarian, brute or distance. For more information about the implementation, see the `RMSD <https://github.com/charnley/rmsd>`_ package.

The reorder option can be used together with the ``-ns`` option, that receives an integer with the number of atoms of the solute. When the ``-ns`` option is used, the script will first superpose the configurations considering only the solute atoms and then reorder considering only the solvent atoms (the atoms in the input that are after the ns atoms). For solute-solvent systems, the use of ``-ns`` is strongly encouraged.

To use an already saved RMSD matrix, specify the file containing the RMSD matrix in the condensed form with the ``-i`` option. The options ``-i`` and ``-od`` are mutually exclusive, since ``-od`` specifies the name of the output RMSD matrix file to be stored.

The ``-p`` flag specifies that pdf plots of some information will be saved.
In this case, the filenames will start with the same name used for the clusters output (specified with the ``-oc`` option). When the option is used, the following is saved to disk:

- A plot with the `multidimensional scaling <http://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling>`_ representation of the RMSD matrix, colored with the clustering information
- The `dendrogram <https://en.wikipedia.org/wiki/Dendrogram>`_
- The cluster classification evolution, that shows how during the trajectory, the configurations were classificated. This might be useful to analyze the quality of your sampling.

If the ``-cc`` option is specified (along with a format supported by OpenBabel) the configurations belonging to the same cluster are superposed and printed to a file.
The superpositions are done considering the `medoid <https://en.wikipedia.org/wiki/Medoid>`_ of the cluster as reference. The medoid is printed as the first structure in the clustered strcuture files.

If you did not consider the hydrogens while building the RMSD matrix, remember to use the ``-n`` option even if with ``-i`` in this case, since the superposition is done considering the flag.

Threading and parallelization
-----------------------------

The ``-np`` option specified the number of processes to be used to calculate the RMSD matrix. 

Since this is the most time consuming task of the clustering, and due to being a embarassingly parallel problem, it was parallelized using a Python `multiprocessing pool <https://docs.python.org/3/library/multiprocessing.html>`_.
The default value for ``-np`` is 4.

When using ``-np`` make sure you also set the correct number of threads for ``numpy``.
If you want to use just the ``multiprocessing`` parallelization (recommended) use the following bash commands to set the number of ``numpy`` threads to one:

.. code-block:: console

	export OMP_NUM_THREADS=1
	export OPENBLAS_NUM_THREADS=1
	export MKL_NUM_THREADS=1
	export VECLIB_MAXIMUM_THREADS=1
	export NUMEXPR_NUM_THREADS=1
