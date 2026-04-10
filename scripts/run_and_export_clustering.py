"""Run wind profile clustering analysis and export results to YAML.

This script loads wind data from various sources (ERA5, FGW lidar, or DOWA),
performs clustering using PCA and K-means, creates visualizations, and exports
the results to YAML format for further use.
"""

import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np

# Add src directory to path for imports
srcPath = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(srcPath))

from wind_profile_clustering.clustering import perform_clustering_analysis
from wind_profile_clustering.plotting import plot_all_results
from wind_profile_clustering.export_profiles_and_probabilities_yml import (
    export_wind_profile_shapes_and_probabilities
)


def main():
    """Run wind profile clustering analysis and export results.
    
    User can configure:
    - DATA_SOURCE: Choose between 'era5', 'fgw_lidar', or 'dowa'
    - N_CLUSTERS: Number of clusters to create (default: 8)
    - SAVE_PLOTS: Save plots as PDF files (default: True)
    """
    # =============================================================================
    # USER CONFIGURATION
    # =============================================================================
    N_CLUSTERS = 8  # Number of clusters to create
    SAVE_PLOTS = True  # Save plots as PDF files in results/ directory
    REF_HEIGHT = 200.0  # Reference height for profile normalization (in meters)
    AWESIO_VALIDATE = True
    
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
            'altitude_range': (10, 500),  # 10-500m above ground
            'years': (2011, 2011)  # Years to include
        }
        data = read_data(config)
        outPrefix = 'era5'
        
        # Prepare metadata for ERA5
        metadata = {
            'name': 'ERA5 Wind Profile Clustering',
            'description': 'Wind profile clustering results derived from ERA5 reanalysis data',
            'note': 'ERA5 reanalysis data processed for wind profile clustering analysis',
            'data_source': 'ERA5',
            'location': {
                'latitude': config['location']['latitude'],
                'longitude': config['location']['longitude']
            },
            'time_range': {
                'start_date': str(data['datetime'][0].astype('datetime64[D]')),
                'end_date': str(data['datetime'][-1].astype('datetime64[D]')),
            },
            'altitude_range': list(config['altitude_range'])
        }
        
    elif DATA_SOURCE == 'fgw_lidar':
        print("Using FGW lidar data...")
        from wind_profile_clustering.read_data.fgw_lidar import read_data
        config = {
            'data_dir': 'data/fgw_lidar',
            'read_raw_data': False  # Set to True to process raw .rtd files
        }
        data = read_data(config)
        outPrefix = 'fgw_lidar'
        
        # Prepare metadata for FGW lidar
        metadata = {
            'name': 'FGW Lidar Wind Profile Clustering',
            'description': 'Wind profile clustering results derived from FGW lidar measurements',
            'note': 'FGW lidar measurement data',
            'data_source': 'FGW_Lidar',
            'location': {
                'latitude': None,  # Update with actual coordinates if available
                'longitude': None
            },
            'time_range': {
                'start_date': str(data['datetime'][0].astype('datetime64[D]')),
                'end_date': str(data['datetime'][-1].astype('datetime64[D]')),
            }           
        }
        
    elif DATA_SOURCE == 'dowa':
        print("Using DOWA model data...")
        from wind_profile_clustering.read_data.dowa import read_data
        config = {
            'data_dir': 'data/dowa',
            'name': 'mmij'  # Options: 'mmij', 'mmc', or use 'coords': (lat, lon)
            # Alternative location specifications:
            # 'coords': (52.85, 3.44)  # By coordinates
            # 'i_lat': i, 'i_lon': j    # By grid indices
            # 'iy': i, 'ix': j          # By 1-based grid indices
        }
        data = read_data(config)
        outPrefix = 'dowa_mmij'
        
        # Prepare metadata for DOWA
        metadata = {
            'name': 'DOWA Wind Profile Clustering',
            'description': 'Wind profile clustering results derived from DOWA model data',
            'note': 'DOWA model data from Maasvlakte Meetmast IJmond location',
            'data_source': 'DOWA',
            'location': {
                'latitude': None,  # Update with actual coordinates if available
                'longitude': None
            },
            'time_range': {
                'start_date': str(data['datetime'][0].astype('datetime64[D]')),
                'end_date': str(data['datetime'][-1].astype('datetime64[D]')),
            },            
        }
        
    else:
        raise ValueError(f"Unknown data source: {DATA_SOURCE}. "
                        "Choose from 'era5', 'fgw_lidar', or 'dowa'.\n"
                        "Run 'python check_data_sources.py' to see available options.")
    
    print(f"Loaded {data['n_samples']} samples from {data['years'][0]} to {data['years'][1]}")
    print(f"Altitude range: {data['altitude'].min():.1f} - {data['altitude'].max():.1f} m")
    print(f"Number of clusters: {N_CLUSTERS}")
    
    # Perform clustering analysis
    results = perform_clustering_analysis(data, N_CLUSTERS, ref_height=REF_HEIGHT)
    
    # Extract results
    processedData = results['processedData']
    processedDataFull = results['processedDataFull']
    res = results['clusteringResults']
    labelsFull = results['labelsFull']
    frequencyClusters = results['frequencyClusters']
    
    # Create all plots
    print("\nGenerating visualizations...")
    plot_all_results(processedData, res, processedDataFull, labelsFull, 
                    frequencyClusters, N_CLUSTERS, savePlots=SAVE_PLOTS)
    
    # Export to YAML
    print("\nExporting results to YAML...")
    outputFile = f'results/wind_profiles_and_probabilities_{outPrefix}.yml'
    prl = res['clusters_feature']['parallel']
    prp = res['clusters_feature']['perpendicular']

    probMatrix = export_wind_profile_shapes_and_probabilities(
        data['altitude'], 
        prl, 
        prp, 
        labelsFull, 
        processedDataFull['normalisation_value'],
        processedDataFull['reference_vector_direction'], 
        processedDataFull['n_samples'],
        N_CLUSTERS, 
        outputFile, 
        metadata=metadata,
        refHeight=REF_HEIGHT,
        validate=AWESIO_VALIDATE
    )
    plt.show()


if __name__ == '__main__':
    main()
