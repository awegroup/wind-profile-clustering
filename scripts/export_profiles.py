"""
Script for exporting wind profiles and probabilities to YAML format.

This script runs the export functionality to generate YAML files containing
wind profile shapes and their probability distributions.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import and run the main function from the export module
from wind_profile_clustering.export_profiles_and_probabilities_yml import (
    export_wind_profile_shapes_and_probabilities
)
from wind_profile_clustering.clustering import cluster_normalized_wind_profiles_pca, predict_cluster
from wind_profile_clustering.preprocess_data import preprocess_data
import numpy as np


def main():
    """
    Main function to export wind profiles and probabilities.
    """
    # =============================================================================
    # DATA SOURCE CONFIGURATION
    # =============================================================================
    # Choose which data source to use by setting DATA_SOURCE to one of:
    # 'era5'      - Use ERA5 reanalysis data from NetCDF files
    # 'fgw_lidar' - Use FGW lidar measurements from CSV file
    # 'dowa'      - Use DOWA model data from NetCDF files
    # =============================================================================
    
    DATA_SOURCE = 'era5'  # Change this to select your data source
    
    # =============================================================================
    # DATA SOURCE SPECIFIC CONFIGURATIONS
    # =============================================================================
    
    if DATA_SOURCE == 'era5':
        print("Using ERA5 reanalysis data...")
        from wind_profile_clustering.read_data.era5 import read_data
        config = {
            'data_dir': 'data/era5',
            'location': {'latitude': 52.0, 'longitude': 4.0},  # Netherlands
            'altitude_range': (0, 500),  # 0-500m above ground
            'years': (2011, 2011)  # Test with just one year
        }
        data = read_data(config)
        out_prefix = 'era5'
        
        # Prepare metadata for ERA5
        metadata = {
            'data_source': 'ERA5',
            'location': {
                'latitude': config['location']['latitude'],
                'longitude': config['location']['longitude']
            },
            'time_range': {
                'start_year': config['years'][0],
                'end_year': config['years'][1],
                'years_included': list(range(config['years'][0], config['years'][1] + 1)),
                'months_included': 'all'  # Will be updated based on available data
            },
            'altitude_range_m': list(config['altitude_range']),  # Convert tuple to list
            'note': 'ERA5 reanalysis data processed for wind profile clustering analysis'
        }
        
    elif DATA_SOURCE == 'fgw_lidar':
        print("Using FGW lidar data...")
        from wind_profile_clustering.read_data.fgw_lidar import read_data
        data = read_data()
        out_prefix = 'fgw_lidar'
        
        # Prepare metadata for FGW lidar
        metadata = {
            'data_source': 'FGW_Lidar',
            'location': {
                'latitude': None,  # Update with actual coordinates if available
                'longitude': None
            },
            'time_range': {
                'start_year': None,  # Update based on data
                'end_year': None,
                'years_included': [],
                'months_included': 'varies'
            },
            'note': 'FGW lidar measurement data'
        }
        
    elif DATA_SOURCE == 'dowa':
        print("Using DOWA model data...")
        from wind_profile_clustering.read_data.dowa import read_data
        # Options for DOWA data:
        # - By name: {'name': 'mmij'} or {'name': 'mmc'}
        # - By coordinates: {'coords': (lat, lon)}
        # - By grid indices: {'i_lat': i, 'i_lon': j} or {'iy': i, 'ix': j}
        data = read_data({'name': 'mmij'})  # Use Maasvlakte Meetmast IJmond location
        out_prefix = 'dowa_mmij'
        
        # Prepare metadata for DOWA
        metadata = {
            'data_source': 'DOWA',
            'location': {
                'latitude': None,  # Update with actual coordinates
                'longitude': None
            },
            'time_range': {
                'start_year': None,  # Update based on data
                'end_year': None,
                'years_included': [],
                'months_included': 'varies'
            },
            'note': 'DOWA model data from Maasvlakte Meetmast IJmond location'
        }
        
    else:
        raise ValueError(f"Unknown data source: {DATA_SOURCE}. "
                        "Choose from 'era5', 'fgw_lidar', or 'dowa'.")
    
    print(f"Loaded {data['n_samples']} samples from {data['years'][0]} to {data['years'][1]}")
    print(f"Altitude range: {data['altitude'].min():.1f} - {data['altitude'].max():.1f} m")
    
    # Process data and perform clustering
    processed_data = preprocess_data(data)
    n_clusters = 8
    res = cluster_normalized_wind_profiles_pca(processed_data['training_data'], n_clusters)
    prl, prp = res['clusters_feature']['parallel'], res['clusters_feature']['perpendicular']

    # Get predictions for full dataset
    processed_data_full = preprocess_data(data, remove_low_wind_samples=False)
    labels, frequency_clusters = predict_cluster(processed_data_full['training_data'], n_clusters,
                                                 res['data_processing_pipeline'].predict, res['cluster_mapping'])

    # Export to YAML
    output_file = f'results/wind_profiles_and_probabilities_{out_prefix}.yml'
    prob_matrix = export_wind_profile_shapes_and_probabilities(
        data['altitude'], prl, prp, labels, processed_data_full['normalisation_value'], 
        processed_data_full['reference_vector_direction'], processed_data_full['n_samples'], 
        n_clusters, output_file, metadata=metadata
    )
    
    print(f"Exported wind profiles and probabilities to: {output_file}")
    print(f"Probability matrix shape: {prob_matrix.shape}")
    print(f"Total probability sum: {np.sum(prob_matrix):.2f}%")
    print(f"Total probability per cluster: {np.sum(prob_matrix, axis=(1,2))}")


if __name__ == '__main__':
    main()
