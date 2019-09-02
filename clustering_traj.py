"""
This script takes a trajectory and based on a minimal RMSD classify the structures in clusters.

The RMSD implementation using the Kabsch algorithm to superpose the molecules is taken from: https://github.com/charnley/rmsd
A very good description of the problem of superposition can be found at http://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures
A very good tutorial on hierachical clustering with scipy can be seen at https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
This script performs agglomerative clustering as suggested in https://stackoverflow.com/questions/31085393/hierarchical-clustering-a-pairwise-distance-matrix-of-precomputed-distances

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
from scipy.spatial.distance import squareform
from sklearn import manifold
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing
import itertools

def get_mol_coords(mol):
  q_all = []
  for atom in mol:
    q_all.append(atom.coords)

  return np.asarray(q_all)


def get_mol_info(mol):
  # table to convert atomic number to symbols
  etab = openbabel.OBElementTable()

  q_atoms = []
  q_all = []
  for atom in mol:
    q_atoms.append(etab.GetSymbol(atom.atomicnum))
    q_all.append(atom.coords)

  return np.asarray(q_atoms), np.asarray(q_all)


def build_distance_matrix(trajfile, noh, reorder, natoms, nprocs):
  # create iterator containing information to compute a line of the distance matrix
  inputiterator = zip(itertools.count(), map(lambda x: get_mol_info(x), pybel.readfile(os.path.splitext(trajfile)[1][1:], trajfile)), itertools.repeat(trajfile), itertools.repeat(noh), itertools.repeat(reorder), itertools.repeat(natoms))

  # create the pool with nprocs processes to compute the distance matrix in parallel
  p = multiprocessing.Pool(processes = nprocs)

  # build the distance matrix in parallel
  ldistmat = p.starmap(compute_distmat_line, inputiterator)

  return np.asarray([x for n in ldistmat if len(n) > 0 for x in n])


def compute_distmat_line(idx1, q_info, trajfile, noh, reorder, nsatoms):
  # unpack q_info tuple
  q_atoms, q_all = q_info

  # initialize distance matrix
  distmat = []

  for idx2, mol2 in enumerate(pybel.readfile(os.path.splitext(trajfile)[1][1:], trajfile)):
    # skip if it's not an element from the superior diagonal matrix
    if idx1 >= idx2:
      continue 

    # arrays for second molecule
    p_atoms, p_all = get_mol_info(mol2)

    # consider the H or not consider depending on option
    if nsatoms:
      # get the number of non hydrogen atoms in the solute to subtract if needed
      if noh:
        natoms = len(np.where(p_atoms[:nsatoms] != 'H')[0])
      else:
        natoms = nsatoms

      if noh:
        not_hydrogens = np.where(p_atoms != 'H')
        P = p_all[not_hydrogens]
        Q = q_all[not_hydrogens]
        Pa = p_atoms[not_hydrogens]
        Qa = q_atoms[not_hydrogens]
      else:
        P = p_all
        Q = q_all
        Pa = p_atoms
        Qa = q_atoms        

      pcenter = rmsd.centroid(P[:natoms])
      qcenter = rmsd.centroid(Q[:natoms])
    elif noh:
      not_hydrogens = np.where(p_atoms != 'H')
      P = p_all[not_hydrogens]
      Q = q_all[not_hydrogens]
      Pa = p_atoms[not_hydrogens]
      Qa = q_atoms[not_hydrogens]
      pcenter = rmsd.centroid(P)
      qcenter = rmsd.centroid(Q)
    else:
      P = p_all
      Q = q_all
      Pa = p_atoms
      Qa = q_atoms
      pcenter = rmsd.centroid(P)
      qcenter = rmsd.centroid(Q)

    # center the coordinates at the origin
    P -= pcenter
    Q -= qcenter

    # generate rotation to superpose the solute configuration
    if nsatoms:
      # center the coordinates at the solute
      P -= rmsd.centroid(Q[:natoms])

      # generate a rotation considering only the solute atoms
      U = rmsd.kabsch(P[:natoms], Q[:natoms])

      # rotate the whole system with this rotation
      P = np.dot(P, U)

      # consider only the solvent atoms in the reorder
      prr = reorder(Qa[natoms:], Pa[natoms:], Q[natoms:], P[natoms:])
      prr += natoms
      prr = np.concatenate((np.arange(natoms),prr))
      Pr = P[prr]
      Pra = Pa[prr]
    # reorder the atoms if necessary
    elif reorder:
      prr = reorder(Qa, Pa, Q, P)
      Pr = P[prr]
      Pra = Pa[prr]
    else:
      Pr = P
      Pra = Pa

    # get the RMSD and store it
    distmat.append(rmsd.kabsch_rmsd(Pr, Q))

  return distmat


def save_clusters_config(trajfile, clusters, distmat, noh, reorder, nsatoms, outbasename, outfmt):

  # complete distance matrix
  sqdistmat = squareform(distmat)
  
  for cnum in range(1,max(clusters)+1):

    # create object to output the configurations
    outfile = pybel.Outputfile(outfmt,outbasename+"_"+str(cnum)+"."+outfmt)

    # creates mask with True only for the members of cluster number cnum
    mask = np.array([1 if i==cnum else 0 for i in clusters],dtype=bool)

    # gets the member with smallest sum of distances from the submatrix
    idx = np.argmin(sum(sqdistmat[:,mask][mask,:]))

    # get list with the members of this cluster only and store medoid
    sublist=[num for (num, cluster) in enumerate(clusters) if cluster==cnum]
    medoid = sublist[idx]

    # get the medoid coordinates
    for idx, mol in enumerate(pybel.readfile(os.path.splitext(trajfile)[1][1:], trajfile)):

      if idx != medoid:
        continue

      # medoid coordinates
      tnatoms = len(mol.atoms)
      q_atoms, q_all = get_mol_info(mol)

      if nsatoms:
        # get the number of non hydrogen atoms in the solute to subtract if needed
        if noh:
          natoms = len(np.where(q_atoms[:nsatoms] != 'H')[0])
        else:
          natoms = nsatoms

        if noh:
          not_hydrogens = np.where(q_atoms != 'H')
          Q = np.copy(q_all[not_hydrogens])
          Qa = np.copy(q_atoms[not_hydrogens])
        else:
          Q = np.copy(q_all)
          Qa = np.copy(q_atoms)

        qcenter = rmsd.centroid(Q[:natoms])
      elif noh:
        not_hydrogens = np.where(q_atoms != 'H')
        Q = np.copy(q_all[not_hydrogens])
        qcenter = rmsd.centroid(Q)
        Qa = np.copy(q_atoms[not_hydrogens])
      else:
        Q = np.copy(q_all)
        qcenter = rmsd.centroid(Q)
        Qa = np.copy(q_atoms)

      # center the coordinates at the origin
      Q -= qcenter

      # write medoid configuration to file (molstring is a xyz string used to generate de pybel mol)
      molstring = str(tnatoms)+"\n"+mol.title.rstrip()+"\n"
      for i, coords in enumerate(q_all - qcenter):
        molstring += q_atoms[i]+"\t"+str(coords[0])+"\t"+str(coords[1])+"\t"+str(coords[2])+"\n"
      rmol = pybel.readstring("xyz", molstring)
      outfile.write(rmol)

      break

    # rotate all the cluster members into the medoid and print them to the .xyz file
    for idx, mol in enumerate(pybel.readfile(os.path.splitext(trajfile)[1][1:], trajfile)):

      if not mask[idx] or idx == medoid:
        continue

      # config coordinates
      p_atoms, p_all = get_mol_info(mol)

      if nsatoms:
        if noh:
          not_hydrogens = np.where(p_atoms != 'H')
          P = np.copy(p_all[not_hydrogens])
          Pa = np.copy(p_atoms[not_hydrogens])
        else:
          P = np.copy(p_all)
          Pa = np.copy(p_atoms)     

        pcenter = rmsd.centroid(P[:natoms])
      elif noh:
        not_hydrogens = np.where(p_atoms != 'H')
        P = np.copy(p_all[not_hydrogens])
        pcenter = rmsd.centroid(P)
        Pa = np.copy(p_atoms[not_hydrogens])
      else:
        P = np.copy(p_all)
        pcenter = rmsd.centroid(P)
        Pa = np.copy(p_atoms)

      # center the coordinates at the origin
      P -= pcenter
      p_all -= pcenter

      # generate rotation to superpose the solute configuration
      if nsatoms:
        # generate a rotation considering only the solute atoms
        U = rmsd.kabsch(P[:natoms], Q[:natoms])

        # rotate the whole system with this rotation
        P = np.dot(P, U)
        p_all = np.dot(p_all, U)

        # consider only the solvent atoms in the reorder
        prr = reorder(Qa[natoms:], Pa[natoms:], Q[natoms:], P[natoms:])
        prr += natoms
        prr = np.concatenate((np.arange(natoms),prr))
        Pr = P[prr]
        Pra = Pa[prr]
      # reorder the atoms if necessary
      elif reorder:
        prr = reorder(Qa, Pa, Q, P)
        Pr = P[prr]
        Pra = Pa[prr]
      else:
        Pr = P
        Pra = Pa

      # generate rotation matrix
      U = rmsd.kabsch(Pr, Q)

      # rotate whole configuration (considering hydrogens even with noh)
      p_all = np.dot(p_all, U)

      # write rotated configuration to file (molstring is a xyz string used to generate de pybel mol)
      molstring = str(tnatoms)+"\n"+mol.title.rstrip()+"\n"
      for i, coords in enumerate(p_all):
        molstring += p_atoms[i]+"\t"+str(coords[0])+"\t"+str(coords[1])+"\t"+str(coords[2])+"\n"
      rmol = pybel.readstring("xyz", molstring)
      outfile.write(rmol)

    # closes the file for the cnum cluster
    outfile.close()


def check_positive(value):
  ivalue = int(value)
  if ivalue <= 0:
       raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
  return ivalue


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run a clustering analysis on a trajectory based on the minimal RMSD obtained with a Kabsch superposition.')
  parser.add_argument('trajectory_file', help='path to the trajectory containing the conformations to be classified')
  parser.add_argument('min_rmsd', help='value of RMSD used to classify structures as similar')
  parser.add_argument('-np', '--nprocesses', metavar='NPROCS', type=check_positive, default=2, help='defines the number of processes used to compute the distance matrix and multidimensional representation (default = 2)')
  parser.add_argument('-n', '--no-hydrogen', action='store_true', help='ignore hydrogens when doing the Kabsch superposition and calculating the RMSD')
  parser.add_argument('-p', '--plot', action='store_true', help='enable the multidimensional scaling and dendrogram plot saving the figures in pdf format (filenames use the same basename of the -oc option)')
  parser.add_argument('-m', '--method', metavar='METHOD', default='average', help="method used for clustering (see valid methods at https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.cluster.hierarchy.linkage.html) (default: average)")
  parser.add_argument('-cc', '--clusters-configurations', metavar='EXTENSION', help='save superposed configurations for each cluster in EXTENSION format (basename based on -oc option)')
  parser.add_argument('-oc', '--outputclusters', default='clusters.dat', metavar='FILE', help='file to store the clusters (default: clusters.dat)')
  parser.add_argument('-e', '--reorder', action='store_true', help='reorder atoms of molecules to lower the RMSD (default: Hungarian)')
  parser.add_argument('--reorder-alg', action='store', default="distance", metavar="METHOD", help='select which reorder algorithm to use; hungarian, brute, distance (default). Warning: brute is VERY slow)')
  parser.add_argument('-ns', '--natoms-solute', metavar="NATOMS", type=int, help='number of solute atoms, to ignore these atoms in the reordering process')

  io_group = parser.add_mutually_exclusive_group()
  io_group.add_argument('-i', '--input', type=argparse.FileType('rb'), metavar='FILE', help='file containing input distance matrix in condensed form')
  io_group.add_argument('-od', '--outputdistmat', metavar='FILE', help='file to store distance matrix in condensed form (default: distmat.dat)')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()

  # check input consistency manually since I did not find a good way to use FileType and have the behavior that I wanted
  if args.method not in ["single","complete","average","weighted","centroid","median","ward"]:
    print("The method you selected with -m (%s) is not valid." % args.method)
    sys.exit(1)

  if args.reorder_alg not in ["hungarian", "brute", "distance"]:
    print("The reorder method you selected with --reorder-method (%s) is not valid." % args.reorder_alg)
    sys.exit(1)

  if args.clusters_configurations:
    if args.clusters_configurations not in ["acr", "adf", "adfout", "alc", "arc", "bgf", "box", "bs", "c3d1", "c3d2", "cac", "caccrt", "cache", "cacint", "can", "car", "ccc", "cdx", "cdxml", "cht", "cif", "ck", "cml", "cmlr", "com", "copy", "crk2d", "crk3d", "csr", "cssr", "ct", "cub", "cube", "dmol", "dx", "ent", "fa", "fasta", "fch", "fchk", "fck", "feat", "fh", "fix", "fpt", "fract", "fs", "fsa", "g03", "g92", "g94", "g98", "gal", "gam", "gamin", "gamout", "gau", "gjc", "gjf", "gpr", "gr96", "gukin", "gukout", "gzmat", "hin", "inchi", "inp", "ins", "jin", "jout", "mcdl", "mcif", "mdl", "ml2", "mmcif", "mmd", "mmod", "mol", "mol2", "molden", "molreport", "moo", "mop", "mopcrt", "mopin", "mopout", "mpc", "mpd", "mpqc", "mpqcin", "msi", "msms", "nw", "nwo", "outmol", "pc", "pcm", "pdb", "png", "pov", "pqr", "pqs", "prep", "qcin", "qcout", "report", "res", "rsmi", "rxn", "sd", "sdf", "smi", "smiles", "sy2", "t41", "tdd", "test", "therm", "tmol", "txt", "txyz", "unixyz", "vmol", "xed", "xml", "xyz", "yob", "zin"]:
      print("The format you selected to save the clustered superposed configurations (%s) is not valid." % args.clusters_configurations)
      sys.exit(1)

  if args.natoms_solute and not args.reorder:
    print("Specifying the number of solute atoms is only useful for the reordering algorithms, continuing anyways..")
    natoms = None
  else:
    natoms = args.natoms_solute

  if args.reorder_alg == "hungarian":
    reorder_alg = rmsd.reorder_hungarian
  elif args.reorder_alg == "distance":
    reorder_alg = rmsd.reorder_distance
  elif args.reorder_alg == "brute":
    reorder_alg = rmsd.reorder_brute

  if not args.reorder:
    reorder_alg = None

  if not args.input:
    if not args.outputdistmat:
      args.outputdistmat = "distmat.dat"

    if os.path.exists(args.outputdistmat):
      exit("File %s already exists, specify a new filename with the -od command option. If you are trying to read the distance matrix from a file, use the -i option" % args.outputdistmat)
    else:
      args.outputdistmat = open(args.outputdistmat,'wb')

  if os.path.exists(args.outputclusters):
    exit("File %s already exists, specify a new filename with the -oc command option" % args.outputclusters)
  else:
    args.outputclusters = open(args.outputclusters,'wb')

  # check if distance matrix will be read from input or calculated
  # if a file is specified, read it (TODO: check if the matrix makes sense)
  if args.input:
    print('\nReading condensed distance matrix from %s\n' % args.input.name)
    distmat = np.loadtxt(args.input)
  # build a distance matrix already in the condensed form
  else:
    print('\nCalculating distance matrix\n')
    distmat = build_distance_matrix(args.trajectory_file, args.no_hydrogen, reorder_alg, natoms, args.nprocesses)
    print('Saving condensed distance matrix to %s\n' % args.outputdistmat.name)
    np.savetxt(args.outputdistmat, distmat, fmt='%.18f')

  # linkage
  print("Starting clustering using '%s' method to join the clusters\n" % args.method)
  Z = hcl.linkage(distmat, args.method)

  # build the clusters and print them to file
  clusters = hcl.fcluster(Z, float(args.min_rmsd), criterion='distance')
  print("Saving clustering classification to %s\n" % args.outputclusters.name)
  np.savetxt(args.outputclusters, clusters, fmt='%d')

  # get the elements closest to the centroid (see https://stackoverflow.com/a/39870085/3254658)
  if args.clusters_configurations:
    print("Writing superposed configurations per cluster to files %s\n" % (os.path.splitext(args.outputclusters.name)[0]+"_confs"+"_*"+"."+args.clusters_configurations))
    save_clusters_config(args.trajectory_file, clusters, distmat, args.no_hydrogen, reorder_alg, natoms, os.path.splitext(args.outputclusters.name)[0]+"_confs", args.clusters_configurations)

  if args.plot:
    # plot evolution with o cluster in trajectory
    plt.figure(figsize=(25, 10))
    plt.plot(range(1,len(clusters)+1), clusters, "o-", markersize=4)
    plt.xlabel('Sample Index')
    plt.ylabel('Cluster classification')
    plt.savefig(os.path.splitext(args.outputclusters.name)[0]+"_evo.pdf", bbox_inches='tight')

    # plot the dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('RMSD')
    hcl.dendrogram(
      Z,
      leaf_rotation=90.,  # rotates the x axis labels
      leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.axhline(float(args.min_rmsd),linestyle='--')
    plt.savefig(os.path.splitext(args.outputclusters.name)[0]+"_dendrogram.pdf", bbox_inches='tight')

    # finds the 2D representation of the distance matrix (multidimensional scaling) and plot it
    plt.figure()
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=666, n_init=6, max_iter=300, eps=1e-3, n_jobs=args.nprocesses)
    coords = mds.fit_transform(squareform(distmat))
    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
    plt.scatter(coords[:, 0], coords[:, 1], marker = 'o', c=clusters, cmap=plt.cm.nipy_spectral)
    plt.savefig(os.path.splitext(args.outputclusters.name)[0]+".pdf", bbox_inches='tight')


  # print the cluster sizes
  print("A total of %d cluster(s) was(were) found.\n" % max(clusters))

  print("A total of %d structures were read from the trajectory. The cluster sizes are:" % len(clusters))
  print("Cluster\tSize")
  labels, sizes = np.unique(clusters, return_counts=True)
  for label, size in zip(labels,sizes):
    print("%d\t%d" % (label,size))

  print()