# Clustering Trajectory
This Python script receives a molecular dynamics or Monte Carlo trajectory (in .pdb, .xyz or any format supported by OpenBabel), finds the minimum RMSD between the structures with the Kabsch algorithm and performs agglomerative clustering (a kind of unsupervised machine learning) to classify similar conformations. 
The script should work both in Python 2 or Python 3, given that all the libraries are available.

What the script does is to calculate the distance (using the minimum RMSD) between each configuration of the trajectory, building a distance matrix (stored in the condensed form).
Notice that calculating the distance matrix might take some time depending on how long your trajectories are and how many atoms there are in each configuration.
The distance matrix can also be read from a file (with the `-i` option) to avoid recalculating it everytime you want to change the linkage method or distance of the clustering.

## Dependencies
The implementation rely on several libraries, so before running the script, make sure you have all of them installed in your Python distribution.
Currently, the following libraries are required:
- [argparse](https://docs.python.org/3/library/argparse.html)
- [NumPy](http://www.numpy.org/)
- [OpenBabel](http://openbabel.org/)
- [RMSD](https://github.com/charnley/rmsd)
- [SciPy](https://www.scipy.org/)

You can install most of those libraries using your package manager or with pip:
```
pip install argparse
pip install numpy
pip install openbabel
pip install rmsd
pip install scipy
```

## Usage
To see all the options run the script with the `-h` command option:
```
python clustering_traj.py -h
```

The only mandatory arguments are the path to the file containing the trajectory (in a format that OpenBabel can read with Pybel) and the maximum RMSD between two configurations for them to be considered of the same cluster.
```
python clustering_traj.py trajectory.xyz 1.0
```

Additional options are available for specifying the input and output files and selecting how the clustering is done.
The possible methods used for the agglomerative clustering are the ones available in the linkage method of SciPy's hierarchical clustering.
A list with the possible methods and the description of each of them can be found [here](https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.cluster.hierarchy.linkage.html).
If the `-n` option is used, the hydrogens are ignored when performing the Kabsch algorithm to find the superposition and calculating the RMSD.

## Output
Some basic information about the size clusters are printed to `STDOUT`.
The number of clusters that were found, as well as the number of members for each cluster are printed in a table.
Below there is an example of how this information is printed:
```
Reading condensed distance matrix from distmat_noh.dat

Saving clustering classification to clusters.dat

A total of 3 cluster(s) was(were) found.

A total of 200 structures were read from the trajectory. The cluster sizes are:
Cluster	Size
1				183
2				6
3				11
```

In the cluster output file (`-oc` option, default filename `clusters.dat`) the classification for each structure in the trajectory is printed.
For example, if the first structure of the trajectory belongs to the cluster number *2*, the second structure belongs to cluster *1*, the third to cluster *2* and so on, the file `clusters.dat` will start with
```
2
1
2
.
.
.
```

If you wish to use the distance matrix file to other uses, bear in mind that the matrix is stored in the condensed form, i.e., only the superior diagonal matrix is printed (not including the diagonal).
It means that if you have `N` structures in your trajectory, your file (specified with `-od` option, default filename `distmat.dat`) will have `N(N-1)/2` lines, with each line representing a distance.
