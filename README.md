# Clustering Trajectory
This Python script receives a molecular dynamics or Monte Carlo trajectory (in .pdb, .xyz or any format supported by OpenBabel), finds the minimum RMSD between the structures with the Kabsch algorithm and performs agglomerative clustering to classify similar conformations. 

What the script does is to calculate the distance (using the minimum RMSD) between each configuration of the trajectory, building a distance matrix (in the condensed form).
The distance matrix can also be read from a file (with the `-i` option) to avoid recalculating it everytime you want to change the linkage method or distance of the clustering.

## Dependencies
The implementation rely on several libraries, so before running the script, make sure you have all of them installed in your Python distribution.
Currently, the following libraries are required:
- [NumPy](http://www.numpy.org/)
- [OpenBabel](http://openbabel.org/)
- [RMSD](https://github.com/charnley/rmsd)
- [SciPy](https://www.scipy.org/)

You can install most of those libraries using your package manager or with pip:
```
pip install numpy
pip install openbabel
pip install rmsd
pip install scipy
```

## Running
To see all the options run the script with the `-h` command option:
```
python clustering_traj.py -h
```

The only mandatory arguments are the path to the file containing the trajectory (in a format that OpenBabel can read with Pybel) and the maximum RMSD between two clusters for them to be considered of the same cluster.
Additional options are available to select the linkage method and specifying the input and output files.
