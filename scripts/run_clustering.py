"""
Main script for running wind profile clustering analysis.

This script loads wind data from various sources (ERA5, FGW lidar, or DOWA),
preprocesses the data, performs clustering using PCA and K-means, and creates
visualizations of the results.
"""

import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from wind_profile_clustering.clustering import cluster_normalized_wind_profiles_pca, predict_cluster
from wind_profile_clustering.plotting import plot_all_results
from wind_profile_clustering.preprocess_data import preprocess_data


def main():
    """
    Main function to run wind profile clustering analysis.
    """
    # =============================================================================
    # DATA SOURCE CONFIGURATION
    # =============================================================================
    # Choose which data source to use by setting DATA_SOURCE to one of:
    # 'era5'      - Use ERA5 reanalysis data from NetCDF files
    # 'fgw_lidar' - Use FGW lidar measurements from CSV file
    # 'dowa'      - Use DOWA model data from NetCDF files
    #
    # TIP: Run 'python check_data_sources.py' to see which sources are available
    # =============================================================================
    
    DATA_SOURCE = 'dowa'  # Change this to select your data source
    
    # =============================================================================
    # DATA SOURCE SPECIFIC CONFIGURATIONS
    # =============================================================================
    
    if DATA_SOURCE == 'era5':
        print("Using ERA5 reanalysis data...")
        from wind_profile_clustering.read_data.era5 import read_data
        config = {
            'data_dir': 'data/era5',
            'location': {'latitude': 52.0, 'longitude': 4.0},  # Netherlands
            'altitude_range': (10, 500),  # 10-500m above ground
            'years': (2010, 2011)  # Limit to 2010-2011 for faster processing
        }
        data = read_data(config)
        
    elif DATA_SOURCE == 'fgw_lidar':
        print("Using FGW lidar data...")
        from wind_profile_clustering.read_data.fgw_lidar import read_data
        data = read_data()
        
    elif DATA_SOURCE == 'dowa':
        print("Using DOWA model data...")
        from wind_profile_clustering.read_data.dowa import read_data
        # Options for DOWA data:
        # - By name: {'name': 'mmij'} or {'name': 'mmc'}
        # - By coordinates: {'coords': (lat, lon)}
        # - By grid indices: {'i_lat': i, 'i_lon': j} or {'iy': i, 'ix': j}
        data = read_data({'name': 'mmij'})  # Use Maasvlakte Meetmast IJmond location
        
    else:
        raise ValueError(f"Unknown data source: {DATA_SOURCE}. "
                        "Choose from 'era5', 'fgw_lidar', or 'dowa'.\n"
                        "Run 'python check_data_sources.py' to see available options.")
    
    print(f"Loaded {data['n_samples']} samples from {data['years'][0]} to {data['years'][1]}")
    print(f"Altitude range: {data['altitude'].min():.1f} - {data['altitude'].max():.1f} m")
    
    # Preprocess data and perform clustering
    processed_data = preprocess_data(data)
    n_clusters = 8
    res = cluster_normalized_wind_profiles_pca(processed_data['training_data'], n_clusters)
    
    # Get predictions for full dataset
    processed_data_full = preprocess_data(data, remove_low_wind_samples=False)
    labels, frequency_clusters = predict_cluster(processed_data_full['training_data'], n_clusters,
                                                 res['data_processing_pipeline'].predict, res['cluster_mapping'])
    
    # Create all plots
    plot_all_results(processed_data, res, processed_data_full, labels, frequency_clusters, n_clusters)
    
    plt.show()


if __name__ == '__main__':
    main()
