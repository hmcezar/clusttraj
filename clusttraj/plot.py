"""Functions to plot the obtained results."""

from sklearn import manifold
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as hcl
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_hex
from matplotlib.ticker import MaxNLocator
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

    # Define a color for the lines
    line_color = (0, 0, 0, 0.5)

    # plot evolution with o cluster in trajectory
    plt.figure(figsize=(10, 6))

    # Set the y-axis to only show integers
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # Increase tick size and font size
    plt.tick_params(axis="both", which="major", direction="in", labelsize=12)

    plt.plot(range(1, len(clusters) + 1), clusters, markersize=4, color=line_color)
    plt.scatter(
        range(1, len(clusters) + 1),
        clusters,
        marker="o",
        c=clusters,
        cmap=plt.cm.nipy_spectral,
    )
    plt.xlabel("Sample Index", fontsize=14)
    plt.ylabel("Cluster classification", fontsize=14)
    plt.savefig(clust_opt.evo_name, bbox_inches="tight")


def plot_dendrogram(
    clust_opt: ClustOptions, clusters: np.ndarray, Z: np.ndarray
) -> None:
    """Plot a dendrogram based on hierarchical clustering.

    Parameters:
        clust_opt (ClustOptions): The options for clustering.
        clusters (np.ndarray): The cluster labels.
        Z (np.ndarray): The linkage matrix.

    Returns:
        None
    """
    # Plot the dendrogram
    plt.figure(figsize=(18, 6))
    plt.title("Hierarchical Clustering Dendrogram", fontsize=20)
    # plt.xlabel("Sample Index", fontsize=14)
    plt.ylabel(r"RMSD ($\AA$)", fontsize=18)
    plt.tick_params(axis="y", labelsize=18)

    # Define a color for the dashed and non-cluster lines
    line_color = (0, 0, 0, 0.5)

    # Add a horizontal line at the minimum RMSD value and set the threshold
    if clust_opt.silhouette_score:
        if isinstance(clust_opt.optimal_cut, (np.ndarray, list)):
            plt.axhline(
                clust_opt.optimal_cut[0], linestyle="--", linewidth=2, color=line_color
            )
            threshold = clust_opt.optimal_cut[0]
        elif isinstance(clust_opt.optimal_cut, (float, np.float32, np.float64)):
            plt.axhline(
                clust_opt.optimal_cut, linestyle="--", linewidth=2, color=line_color
            )
            threshold = clust_opt.optimal_cut
        else:
            raise ValueError("optimal_cut must be a float or np.ndarray")
    else:
        plt.axhline(clust_opt.min_rmsd, linestyle="--", linewidth=2, color=line_color)
        threshold = clust_opt.min_rmsd

    # Use the 'nipy_spectral' cmap to color the dendrogram
    unique_clusters = np.unique(clusters)
    cmap = cm.get_cmap("nipy_spectral", len(unique_clusters))
    colors = [to_hex(cmap(i)) for i in range(cmap.N)]

    hierarchy.set_link_color_palette(colors)

    # Plot the dendrogram
    hcl.dendrogram(
        Z,
        # leaf_rotation=90.0,  # Rotates the x axis labels
        # leaf_font_size=8.0,  # Font size for the x axis labels
        no_labels=True,
        color_threshold=threshold,
        above_threshold_color=line_color,
    )

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

    # Set the figure size
    plt.figure(figsize=(6, 6))

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

    plt.title("MDS Visualization", fontsize=14)

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
    cmap = cm.get_cmap("nipy_spectral", len(unique_clusters))
    colors = [cmap(i) for i in range(len(unique_clusters))]

    # Set the figure size
    plt.figure(figsize=(6, 6))

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
