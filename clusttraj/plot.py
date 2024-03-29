"""Functions to plot the obtained results."""

from sklearn import manifold
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as hcl
import matplotlib.pyplot as plt
import numpy as np
from .io import ClustOptions


def plot_clust_evo(clust_opt: ClustOptions, clusters: np.ndarray) -> None:
    """Plot the evolution of cluster classification over the given samples.

    Args:
        clust_opt (ClustOptions): The clustering options.
        clusters (np.ndarray): The cluster classifications for each sample.

    Returns:
        None
    """
    # plot evolution with o cluster in trajectory
    plt.figure(figsize=(25, 10))
    plt.plot(range(1, len(clusters) + 1), clusters, "o-", markersize=4)
    plt.xlabel("Sample Index")
    plt.ylabel("Cluster classification")
    plt.savefig(clust_opt.evo_name, bbox_inches="tight")


def plot_dendrogram(clust_opt: ClustOptions, Z: np.ndarray) -> None:
    """Plot a dendrogram based on hierarchical clustering.

    Parameters:
        clust_opt (ClustOptions): The options for clustering.
        Z (np.ndarray): The linkage matrix.

    Returns:
        None
    """
    # Plot the dendrogram
    plt.figure(figsize=(25, 10))
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel(r"RMSD ($\AA$)")

    hcl.dendrogram(
        Z,
        leaf_rotation=90.0,  # Rotates the x axis labels
        leaf_font_size=8.0,  # Font size for the x axis labels
    )

    # Add a horizontal line at the minimum RMSD value
    if clust_opt.silhouette_score:
        if isinstance(clust_opt.optimal_cut, (np.ndarray, list)):
            plt.axhline(clust_opt.optimal_cut[0], linestyle="--")
        elif isinstance(clust_opt.optimal_cut, (float, np.float32, np.float64)):
            plt.axhline(clust_opt.optimal_cut, linestyle="--")
        else:
            raise ValueError("optimal_cut must be a float or np.ndarray")
    else:
        plt.axhline(clust_opt.min_rmsd, linestyle="--")

    # Save the dendrogram to a file
    plt.savefig(clust_opt.dendrogram_name, bbox_inches="tight")


def plot_mds(clust_opt: ClustOptions, clusters: np.ndarray, distmat: np.ndarray) -> None:
    """Plot the multidimensional scaling (MDS) of the distance matrix.

    Args:
        clust_opt (ClustOptions): The clustering options.
        clusters (np.ndarray): The cluster labels.
        distmat (np.ndarray): The distance matrix.

    Returns:
        None
    """
    # Create a new figure
    plt.figure()

    # Initialize the MDS model
    mds = manifold.MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=666,
        n_init=3,
        max_iter=200,
        eps=1e-3,
        n_jobs=clust_opt.n_workers,
        normalized_stress="auto",
    )

    # Perform MDS and get the 2D representation
    coords = mds.fit_transform(squareform(distmat))

    # Configure tick parameters
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

    # Scatter plot the coordinates with cluster colors
    plt.scatter(
        coords[:, 0], coords[:, 1], marker="o", c=clusters, cmap=plt.cm.nipy_spectral
    )

    plt.title("MDS Visualization")

    # Save the plot
    plt.savefig(clust_opt.mds_name, bbox_inches="tight")


def plot_tsne(
    clust_opt: ClustOptions, clusters: np.ndarray, distmat: np.ndarray
) -> None:
    """Plot the t-distributed Stochastic Neighbor Embedding 2D plot of the clustering.

    Args:
        clust_opt (ClustOptions): The clustering options.
        clusters (np.ndarray): The cluster labels.
        distmat (np.ndarray): The distance matrix.

    Returns:
        None
    """

    # Initialize the tSNE model
    tsne = manifold.TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        random_state=666,
        n_jobs=clust_opt.n_workers,
    )

    # Perform the t-SNE and get the 2D representation
    coords = tsne.fit_transform(squareform(distmat))

    # Define a list of unique colors for each cluster
    unique_clusters = np.unique(clusters)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    # Create a new figure
    plt.figure()

    # Configure tick parameters
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

    # Create a scatter plot with different colors for each cluster
    for i, cluster in enumerate(unique_clusters):
        cluster_data = coords[clusters == cluster]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=colors[i])

    plt.title("t-SNE Visualization")

    # Save the plot
    plt.savefig(clust_opt.mds_name[:-7] + "tsne.pdf", bbox_inches="tight")
