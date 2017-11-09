"""
Script used to find clusters using RMSD.
The RMSD implementation using the Kabsch algorithm to superpose the molecules is taken from: https://github.com/charnley/rmsd
To install this RMSD package, use pip (more at https://github.com/charnley/rmsd).
A very good description of the problem of superposition can be found at http://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures
A very good tutorial on hierachical clustering with scipy can be seen at https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
Performs hierachic cluster as suggested in https://stackoverflow.com/questions/31085393/hierarchical-clustering-a-pairwise-distance-matrix-of-precomputed-distances

This script takes a trajectory and based on a minimal RMSD classify the structures in clusters.

Author: Henrique Musseli Cezar
Date: NOV/2017
"""

import os
import sys
import argparse
import numpy as np
import rmsd
import pybel
import openbabel
import scipy.cluster.hierarchy as hcl

def check_file_exists(parser, arg):
  if os.path.exists(arg):
    parser.error("file %s already exists, specify a new filename with the right command option" % arg)
  else:
    return open(arg,'wb')

def build_distance_matrix(trajfile, noh):
  # table to convert atomic number to symbols
  etab = openbabel.OBElementTable()

  # initialize distance matrix
  distmat = []

  # start building the distance matrix
  for idx1, mol1 in enumerate(pybel.readfile(os.path.splitext(trajfile)[1][1:], trajfile)):

    # arrays for first molecule
    natoms = len(mol1.atoms)
    q_atoms = []
    q_all = []
    for atom in mol1:
      q_atoms.append(etab.GetSymbol(atom.atomicnum))
      q_all.append(atom.coords)
    q_atoms = np.array(q_atoms)
    q_all = np.array(q_all)

    for idx2, mol2 in enumerate(pybel.readfile(os.path.splitext(trajfile)[1][1:], trajfile)):
      # skip if it's not an element from the superior diagonal matrix
      if idx1 >= idx2:
        continue 

      # arrays for second molecule
      p_atoms = []
      p_all = []
      for atom in mol2:
        p_atoms.append(etab.GetSymbol(atom.atomicnum))
        p_all.append(atom.coords)
      p_atoms = np.array(p_atoms)
      p_all = np.array(p_all)

      if np.count_nonzero(p_atoms != q_atoms):
        exit("Atoms not in the same order")

      # consider the H or not consider depending on option
      if noh:
        not_hydrogens = np.where(p_atoms != 'H')
        P = p_all[not_hydrogens]
        Q = q_all[not_hydrogens]
      else:
        P = p_all
        Q = q_all

      # center the coordinates at the origin
      P -= rmsd.centroid(P)
      Q -= rmsd.centroid(Q)

      # get the RMSD and store it
      distmat.append(rmsd.kabsch_rmsd(P, Q))

  return np.asarray(distmat)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Run a clustering analysis on a trajectory based on the minimal RMSD obtained with a Kabsch superposition.")
  parser.add_argument("trajectory_file", help="path to the trajectory containing the conformations to be classified")
  parser.add_argument("min_rmsd", help="value of RMSD used to classify structures as similar")
  parser.add_argument('-n', '--no-hydrogen', action='store_true', help='ignore hydrogens when doing the Kabsch superposition and calculating the RMSD')
  parser.add_argument('-m', '--method', metavar='METHOD', default='ward', help="method used for clustering (see valid methods at https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.cluster.hierarchy.linkage.html) (default: ward)")
  parser.add_argument('-oc', '--outputclusters', default='clusters.dat', metavar='FILE', help='file to store the clusters (default: clusters.dat)')

  io_group = parser.add_mutually_exclusive_group()
  io_group.add_argument('-i', '--input', type=argparse.FileType('rb'), metavar='FILE', help='file containing input distance matrix in condensed form')
  io_group.add_argument('-od', '--outputdistmat', metavar='FILE', help='file to store distance matrix in condensed form (default: distmat.dat)')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()

  # check input consistency manually since I did not find a good way to use FileType and have the behavior that I wanted
  if not args.input:
    if not args.outputdistmat:
      args.outputdistmat = "distmat.dat"

    if os.path.exists(args.outputdistmat):
      exit("file %s already exists, specify a new filename with the -od command option" % args.outputdistmat)
    else:
      args.outputdistmat = open(args.outputdistmat,'wb')

  if os.path.exists(args.outputclusters):
    exit("file %s already exists, specify a new filename with the -oc command option" % args.outputclusters)
  else:
    args.outputclusters = open(args.outputclusters,'wb')

  # check if distance matrix will be read from input or calculated
  # if a file is specified, read it (TODO: check if the matrix makes sense)
  if (args.input):
    print('\nReading condensed distance matrix from %s\n' % args.input.name)
    distmat = np.loadtxt(args.input)
  # build a distance matrix already in the condensed form
  else:
    print('\nCalculating distance matrix\n')
    distmat = build_distance_matrix(args.trajectory_file, args.no_hydrogen)
    print('Saving condensed distance matrix to %s\n' % args.outputdistmat.name)
    np.savetxt(args.outputdistmat, distmat, fmt='%.18f')

  # linkage
  Z = hcl.linkage(distmat, args.method)

  # build the clusters and print them to file
  clusters = hcl.fcluster(Z, float(args.min_rmsd), criterion='distance')
  print("Saving clustering classification to %s\n" % args.outputclusters.name)
  np.savetxt(args.outputclusters, clusters, fmt='%d')

  # print the cluster sizes
  print("A total of %d cluster(s) was(were) found.\n" % max(clusters))

  print("A total of %d structures were read from the trajectory. The cluster sizes are:" % len(clusters))
  print("Cluster\tSize")
  labels, sizes = np.unique(clusters, return_counts=True)
  for label, size in zip(labels,sizes):
    print("%d\t%d" % (label,size))

  print()