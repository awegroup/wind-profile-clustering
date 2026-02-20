"""
Wind Profile Clustering Package

A package for clustering wind profile data using PCA and K-means clustering.
"""

from .clustering import (
    cluster_normalized_wind_profiles_pca,
    predict_cluster
)

from .plotting import (
    plot_wind_profile_shapes,
    plot_bars,
    visualise_patterns,
    projection_plot_of_clusters
)

__version__ = '0.1.0'
