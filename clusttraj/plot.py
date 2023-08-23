from sklearn import manifold
from scipy.spatial.distance import squareform
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_clust_evo(clust_opt, clusters):
    # plot evolution with o cluster in trajectory
    plt.figure(figsize=(25, 10))
    plt.plot(range(1, len(clusters) + 1), clusters, "o-", markersize=4)
    plt.xlabel("Sample Index")
    plt.ylabel("Cluster classification")
    plt.savefig(clust_opt.evo_name, bbox_inches="tight")


def plot_dendrogram(clust_opt, Z):
    # plot the dendrogram
    plt.figure(figsize=(25, 10))
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("RMSD")
    hcl.dendrogram(
        Z,
        leaf_rotation=90.0,  # rotates the x axis labels
        leaf_font_size=8.0,  # font size for the x axis labels
    )
    plt.axhline(clust_opt.min_rmsd, linestyle="--")
    plt.savefig(clust_opt.dendrogram_name, bbox_inches="tight")


def plot_mds(clust_opt, distmat):
    # finds the 2D representation of the distance matrix (multidimensional scaling) and plot it
    plt.figure()
    mds = manifold.MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=666,
        n_init=3,
        max_iter=200,
        eps=1e-3,
        n_jobs=clust_opt.n_workers,
    )
    coords = mds.fit_transform(squareform(distmat))
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    plt.scatter(
        coords[:, 0], coords[:, 1], marker="o", c=clusters, cmap=plt.cm.nipy_spectral
    )
    plt.savefig(clust_opt.mds_name, bbox_inches="tight")
