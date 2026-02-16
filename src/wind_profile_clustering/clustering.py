"""
Clustering functions for wind profile analysis.

This module contains functions for clustering normalized wind profiles using
PCA and K-means clustering algorithms.
"""

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import numpy as np


def cluster_normalized_wind_profiles_pca(training_data, n_clusters, n_pcs=5, reorder=None):
    """
    Cluster normalized wind profiles using PCA and K-means.

    Args:
        training_data (ndarray): Preprocessed wind profile data.
        n_clusters (int): Number of clusters to create.
        n_pcs (int): Number of principal components to use. Defaults to 5.
        reorder (list or ndarray): Optional reordering of clusters. Defaults to None.

    Returns:
        dict: Dictionary containing clustering results with keys:
            - clusters_pc: Cluster centers in PC space.
            - clusters_feature: Cluster centers in original feature space.
            - frequency_clusters: Frequency of each cluster.
            - sample_labels: Cluster label for each sample.
            - fit_inertia: K-means inertia value.
            - data_processing_pipeline: Pipeline for prediction.
            - pca: Fitted PCA object.
            - training_data_pc: Training data in PC space.
            - cluster_mapping: Mapping from original to reordered clusters.
            - pc_explained_variance: Explained variance of each PC.
    """
    # Use the (prepocessed) data to find the set of profile shapes that represent the variation in the data the best.
    n_samples = len(training_data)

    pca = PCA(n_components=n_pcs)
    training_data_pc = pca.fit_transform(training_data)
    print("Components reduced from {} to {}.".format(training_data.shape[1], pca.n_components_))

    cluster_model = KMeans(n_clusters=n_clusters, random_state=0).fit(training_data_pc)

    mean_inertia_fit = cluster_model.inertia_/n_samples
    mean_distance = mean_inertia_fit**.5
    print("Mean distance: {:.3f}".format(mean_distance))

    # Determine how much samples belong to each cluster.
    freq = np.zeros(n_clusters)
    for l in cluster_model.labels_:  # Labels: Index of the cluster each sample belongs to.
        freq[l] += 100. / n_samples

    # By default order the clusters on their size.
    plot_order = np.array(sorted(range(n_clusters), key=freq.__getitem__, reverse=True))
    if reorder:
        plot_order = plot_order[reorder]
    clusters_pc = cluster_model.cluster_centers_[plot_order, :]
    freq = freq[plot_order]
    labels = np.zeros(n_samples).astype(int)
    for i_new, i_old in enumerate(plot_order):
        labels[cluster_model.labels_ == i_old] = i_new

    # Retrieve the mean cluster shapes in original coordinate system.
    clusters_feature = pca.inverse_transform(clusters_pc)
    n_altitudes = training_data.shape[1]//2

    res = {
        'clusters_pc': clusters_pc,
        'clusters_feature': {
            'parallel': clusters_feature[:, :n_altitudes],
            'perpendicular': clusters_feature[:, n_altitudes:]
        },
        'frequency_clusters': freq,
        'sample_labels': labels,
        'fit_inertia': cluster_model.inertia_,
        'data_processing_pipeline': make_pipeline(pca, cluster_model),
        'pca': pca,
        'training_data_pc': training_data_pc,
        'cluster_mapping': plot_order,
        'pc_explained_variance': pca.explained_variance_,
    }
    return res


def predict_cluster(training_data, n_clusters, predict_fun, cluster_mapping):
    """
    Predict cluster labels for new data using a trained model.

    Args:
        training_data (ndarray): Preprocessed wind profile data.
        n_clusters (int): Number of clusters.
        predict_fun (callable): Prediction function from trained pipeline.
        cluster_mapping (ndarray): Mapping from original to reordered clusters.

    Returns:
        tuple: Tuple containing:
            - labels (ndarray): Predicted cluster labels.
            - frequency_clusters (ndarray): Frequency of each cluster.
    """
    n_samples = len(training_data)
    labels_unarranged = predict_fun(training_data)
    labels = np.zeros(n_samples).astype(int)
    for i_new, i_old in enumerate(cluster_mapping):
        labels[labels_unarranged == i_old] = i_new

    # Determine how much samples belong to each cluster.
    frequency_clusters = np.zeros(n_clusters)
    for l in labels:  # Labels: Index of the cluster each sample belongs to.
        frequency_clusters[l] += 100. / n_samples

    return labels, frequency_clusters
