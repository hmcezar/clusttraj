Examples
========

Before following the steps presented in this section, make sure to install the ``clusttraj`` package as presented in :doc:`install`.

Clustering of water molecules
*****************************

Here we perform the clustering of water molecules from a molecular dynamics simulation. The ``h2o_traj.xyz`` file has the molecular trajectory of 5 water molecules:

.. code-block:: console

	╰─○ head h2o_traj.xyz

	15
	Frame 1
	       O     5.79224   23.34640   11.44660
	       H     4.98409   23.01319   11.83659
	       H     5.84046   22.91839   10.59179
	       O     5.59517   22.31815    8.86907
	       H     5.09795   21.51188    8.73143
	       H     6.50540   22.06908    8.70880
	       O     2.78349   22.26232   10.73374
	       H     2.46676   23.16482   10.69619


To perform the clustering procedure we need to provide the trajectory file and the RMSD threshold distance. This cutoff distance establishs the maximum accepted distance between clusters and can be determined in two ways.

Manual threshold
^^^^^^^^^^^^^^^^

We can fix the maximum RMSD deviation between units in the same cluster up to a certain number, `e.g.`, 2.0 Angstrons:

.. code-block:: console
	
	python -m clusttraj h2o_traj.xyz -rmsd 2.0
	
As a result, we obtained 4 output files, `i.e.`, ``distmat.npy``, ``clusters.dat``, ``clusters.out`` and ``clusttraj.log``.

- ``distmat.npy`` file has the condensed distance matrix in the ``numpy`` file format.

- ``clusters.dat`` file has the labels of each configuration in the trajectory file.

.. code-block:: console

	╰─○ head clusters.dat
	3
	3
	1
	2
	1
	1
	2
	1
	1
	3

- ``clusters.out`` file has the simulation details and the cluster sizes.


.. code-block:: console
	
	╰─○ cat clusters.out

	Full command: /Users/Rafael/Coisas/Doutorado/clusttraj/clusttraj/clusttraj/__main__.py h2o_traj.xyz -rmsd 2.0 -i distmat.npy

	Clusterized from trajectory file: h2o_traj.xyz
	Method: average
	RMSD criterion: 2.0
	Ignoring hydrogens?: False

	Distance matrix was read from: distmat.npy
	The classification of each configuration was written in: clusters.dat
	A total 100 snapshots were read and 3 cluster(s) was(were) found.
	The cluster sizes are:
	Cluster	Size
	1	44
	2	22
	3	34

- ``clusttraj.log`` file has the system output and the cluster sizes.

.. code-block:: console

	╰─○ cat clusttraj.log
	2023-09-30 16:39:02,100 INFO     [distmat.py:28] <get_distmat> Reading condensed distance matrix from distmat.npy

	2023-09-30 16:39:02,102 INFO     [classify.py:97] <classify_structures> Clustering using 'average' method to join the clusters

	2023-09-30 16:39:02,103 INFO     [classify.py:105] <classify_structures> Saving clustering classification to clusters.dat

	2023-09-30 16:39:02,105 INFO     [main.py:75] <main> A total 100 snapshots were read and 3 cluster(s) was(were) found.
	The cluster sizes are:
	Cluster	Size
	1	44
	2	22
	3	34

Automatic threshold
^^^^^^^^^^^^^^^^^^^

Instead of manually fixing the maximum RMSD, we can run the ``-ss`` flag to determine the threshold as the value that maximizes the silhouette coefficient. The coefficient varies between -1 and 1, such that higher values indicate a better clustering procedure. Further details can be found `here <LINK-DO-PAPER>`_.

.. code-block:: console

	python -m clusttraj h2o_traj.xyz -ss -i distmat.npy -p

Since we already computed the distance matrix, we can provide it as input using the ``-i`` flag. Additionally, the ``-p`` flag generates 3 new output files for visualization.

- ``clusters.pdf`` plots the multidimensional scaling (MDS) of the distance matrix.

.. image:: images/average_full_mds.pdf
	:width: 300pt

- ``clusters_dendrogram.pdf`` plots the hierarchical clustering dendrogram.

.. image:: images/average_full_dend.pdf
	:width: 300pt

- ``clusters_evo.pdf`` plots the evolution of cluster populations during the simulation.

.. image:: images/average_full_evo.pdf
	:width: 300pt

The highest silhouette score is printed in the ``clusttraj.log`` file, along with the corresponding RMSD threshold:

.. code-block:: console
	
	╰─○ cat clusttraj.log
	2023-09-30 17:04:14,908 INFO     [distmat.py:28] <get_distmat> Reading condensed distance matrix from distmat.npy

	2023-09-30 17:04:14,916 INFO     [classify.py:27] <classify_structures_silhouette> Clustering using 'average' method to join the clusters

	2023-09-30 17:04:15,064 INFO     [classify.py:61] <classify_structures_silhouette> Highest silhouette score: 0.21741836027295444

	2023-09-30 17:04:15,065 INFO     [classify.py:65] <classify_structures_silhouette> The following RMSD threshold values yielded the same optimial silhouette score: 2.160840752745414, 2.2608407527454135

	2023-09-30 17:04:15,065 INFO     [classify.py:68] <classify_structures_silhouette> The smallest RMSD of 2.160840752745414 has been adopted

	2023-09-30 17:04:15,065 INFO     [classify.py:76] <classify_structures_silhouette> Saving clustering classification to clusters.dat

	2023-09-30 17:04:21,562 INFO     [main.py:75] <main> A total 100 snapshots were read and 2 cluster(s) was(were) found.
	The cluster sizes are:
	Cluster	Size
	1	44
	2	56 

To determine the optimal threshold the silhouette coefficient is computed for all values in in the `linkage matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_ with the default step of 0.1. In this case more than one value yields the same optimal threshold (2.16 and 2.26), and the smallest one is adopted to enhance the within cluster similarity. 

Working with distance methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To investigate the performance of different cluster distance methods we can use the ``-m`` flag.

Ward
++++

In the case of following the Ward variance minimization algorithm:

.. code-block:: console
	
	python -m clusttraj -ss -i distmat.npy -p -m ward -f

In this approach the ``ward`` method is adopted instead of the default ``average`` method. The ``-f`` flag is also included to force the output overwrite with the new data. From the log file:

.. code-block:: console

	╰─○ cat clusttraj.log
	2023-09-30 18:07:18,729 INFO     [distmat.py:28] <get_distmat> Reading condensed distance matrix from distmat.npy

	2023-09-30 18:07:18,730 INFO     [classify.py:27] <classify_structures_silhouette> Clustering using 'ward' method to join the clusters

	2023-09-30 18:07:18,943 INFO     [classify.py:61] <classify_structures_silhouette> Highest silhouette score: 0.23037242401157293

	2023-09-30 18:07:18,943 INFO     [classify.py:65] <classify_structures_silhouette> The following RMSD threshold values yielded the same optimial silhouette score: 6.060840752745413, 6.160840752745413, 6.260840752745413, 6.360840752745413, 6.460840752745413, 6.5608407527454125, 6.660840752745413, 6.760840752745413, 6.860840752745412, 6.960840752745413, 7.0608407527454125, 7.160840752745413, 7.260840752745413, 7.360840752745412, 7.460840752745413, 7.5608407527454125, 7.660840752745413, 7.760840752745413, 7.860840752745412, 7.960840752745413, 8.060840752745412, 8.160840752745411, 8.260840752745413, 8.360840752745412, 8.460840752745412, 8.560840752745412, 8.660840752745411, 8.760840752745413, 8.860840752745412, 8.960840752745412, 9.060840752745412, 9.160840752745413, 9.260840752745413, 9.360840752745412, 9.460840752745412, 9.560840752745412, 9.660840752745411, 9.760840752745413, 9.860840752745412, 9.960840752745412, 10.060840752745412, 10.160840752745411, 10.260840752745413, 10.360840752745412, 10.460840752745412, 10.560840752745412, 10.660840752745411, 10.760840752745413

	2023-09-30 18:07:18,943 INFO     [classify.py:68] <classify_structures_silhouette> The smallest RMSD of 6.060840752745413 has been adopted

	2023-09-30 18:07:18,943 INFO     [classify.py:76] <classify_structures_silhouette> Saving clustering classification to clusters.dat

	2023-09-30 18:07:25,197 INFO     [main.py:75] <main> A total 100 snapshots were read and 2 cluster(s) was(were) found.
	The cluster sizes are:
	Cluster	Size
	1	46
	2	54


The ``ward`` method also resulted in two clusters with similar populations (46/54 vs 44/56) and with higher silhouette coefficient (0.230 vs 0.217). On the other hand, the smallest RMSD threshold is 6.06, indicating higher deviation between the geometries in each cluster.

Median
++++++

To adopt the ``median`` method we can run:

.. code-block:: console

	python -m clusttraj h2o_traj.xyz -ss -i distmat.npy -p -m median -f 

	╰─○ cat clusttraj.log 
	2023-09-30 18:23:54,842 INFO     [distmat.py:28] <get_distmat> Reading condensed distance matrix from distmat.npy

	2023-09-30 18:23:54,843 INFO     [classify.py:27] <classify_structures_silhouette> Clustering using 'median' method to join the clusters

	2023-09-30 18:23:54,870 INFO     [classify.py:61] <classify_structures_silhouette> Highest silhouette score: 0.07527635729544939

	2023-09-30 18:23:54,870 INFO     [classify.py:65] <classify_structures_silhouette> The following RMSD threshold values yielded the same optimial silhouette score: 1.8608407527454136, 1.9608407527454137, 2.060840752745414

	2023-09-30 18:23:54,870 INFO     [classify.py:68] <classify_structures_silhouette> The smallest RMSD of 1.8608407527454136 has been adopted

	2023-09-30 18:23:54,870 INFO     [classify.py:76] <classify_structures_silhouette> Saving clustering classification to clusters.dat

	2023-09-30 18:24:00,293 INFO     [main.py:75] <main> A total 100 snapshots were read and 2 cluster(s) was(were) found.
	The cluster sizes are:
	Cluster	Size
	1	99
	2	1


In this case the highest silhouette score of 0.075 indicates that the points are located near the edge of the clusters. The distribution of population among the 2 clusters (1/99) also indicates the limitations of the method. Finally, visual inspection of the dendrogram shows anomalous behavior.

.. image:: images/anomalous_dend.pdf
	:width: 300pt

.. .. raw:: html

.. 	<iframe src='/Users/Rafael/Coisas/Doutorado/clusttraj/clusttraj/docs/build/html/_images/anomalous_dendrogram.pdf' width="100%" height="500"></iframe>

The reader is encouraged to verify that the addition of ``-odl`` for `optimal visualization <https://academic.oup.com/bioinformatics/article/17/suppl_1/S22/261423?login=true>`_ flag cannot avoid the dendrogram crossings.


Accouting for molecule permutation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As an attempt to avoid separating similar configurations due to permutation of identical molecules, we can reorder the atoms using the ``-e`` flag. 

.. code-block:: console

	python -m clusttraj h2o_traj.xyz -ss -p -m average -e -f

For this system the reordering compromised the statistical quality of the clustering. The number of clusters was increased from 2 to 35 while the optimal silhouette score was reduced from 0.217 to 0.119:

.. code-block:: console

	╰─○ cat clusttraj.log 
	2023-10-02 19:53:20,618 INFO     [distmat.py:34] <get_distmat> Calculating distance matrix using 4 threads

	2023-10-02 19:54:00,821 INFO     [distmat.py:38] <get_distmat> Saving condensed distance matrix to distmat.npy

	2023-10-02 19:54:00,823 INFO     [classify.py:27] <classify_structures_silhouette> Clustering using 'average' method to join the clusters

	2023-10-02 19:54:00,855 INFO     [classify.py:61] <classify_structures_silhouette> Highest silhouette score: 0.11873407875769024

	2023-10-02 19:54:00,856 INFO     [classify.py:71] <classify_structures_silhouette> Optimal RMSD threshold value: 1.237013337787396

	2023-10-02 19:54:00,856 INFO     [classify.py:76] <classify_structures_silhouette> Saving clustering classification to clusters.dat

	2023-10-02 19:54:06,676 INFO     [main.py:75] <main> A total 100 snapshots were read and 35 cluster(s) was(were) found.
	The cluster sizes are:
	Cluster	Size
	1	2
	2	4
	3	3
	4	1
	5	1
	6	1
	7	2
	8	2
	9	3
	10	2
	11	7
	12	3
	13	7
	14	7
	15	3
	16	5
	17	4
	18	3
	19	2
	20	4
	21	2
	22	3
	23	3
	24	1
	25	2
	26	3
	27	2
	28	1
	29	2
	30	2
	31	5
	32	4
	33	2
	34	1
	35	1

This functionality is especially useful in the case of solvated systems. In our case, we can treat one water molecule as the solute and the others as solvent. For example, considering the first water molecule as the solute:

.. code-block:: console

	python -m clusttraj h2o_traj.xyz -ss -p -m average -e -f -ns 3

The number of solvent atoms must be specified using the ``-ns`` flag, and as a result we managed to increase the silhouette coefficient to 0.247 with a significant change in the cluster populations:

.. code-block:: console

	╰─○ cat clusttraj.log 
	2023-10-02 20:13:52,041 INFO     [distmat.py:38] <get_distmat> Saving condensed distance matrix to distmat.npy

	2023-10-02 20:13:52,044 INFO     [classify.py:27] <classify_structures_silhouette> Clustering using 'average' method to join the clusters

	2023-10-02 20:13:52,101 INFO     [classify.py:61] <classify_structures_silhouette> Highest silhouette score: 0.24735123044958368

	2023-10-02 20:13:52,102 INFO     [classify.py:65] <classify_structures_silhouette> The following RMSD threshold values yielded the same optimial silhouette score: 3.035586843407412, 3.135586843407412, 3.235586843407412, 3.335586843407412

	2023-10-02 20:13:52,102 INFO     [classify.py:68] <classify_structures_silhouette> The smallest RMSD of 3.035586843407412 has been adopted

	2023-10-02 20:13:52,102 INFO     [classify.py:76] <classify_structures_silhouette> Saving clustering classification to clusters.dat

	2023-10-02 20:13:57,498 INFO     [main.py:75] <main> A total 100 snapshots were read and 2 cluster(s) was(were) found.
	The cluster sizes are:
	Cluster	Size
	1	3
	2	97

Final Kabsch rotation
^^^^^^^^^^^^^^^^^^^^^

We can also add a final Kabsch rotation to minimize the RMSD after reordering the solvent atoms:

.. code-block:: console

	python -m clusttraj h2o_traj.xyz -ss -p -m average -e -f -ns 3 --final-kabsch

For this system no significant changes were observed, as the silhouette coefficient and cluster populations remain almost identical.

Removing hydrogen atoms
^^^^^^^^^^^^^^^^^^^^^^^




