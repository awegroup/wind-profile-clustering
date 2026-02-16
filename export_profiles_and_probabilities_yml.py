"""
This script exports wind profile shapes and their probability distributions to a YAML file.

The main function, `export_wind_profile_shapes_and_probabilities`, processes wind profile data, applies normalization, and calculates probability distributions for each cluster. The output includes:

- Metadata about the clustering process and data source.
- A single list of altitude levels shared across all clusters.
- Normalized wind profile components (u and v) for each cluster.
- Probability distributions of wind speeds for each cluster.

The script supports multiple data sources, including ERA5, FGW lidar, and DOWA, and allows for flexible configuration of input data and clustering parameters.

Usage:
- Run the script directly to process and export wind profiles for a specified data source.
- Import the `export_wind_profile_shapes_and_probabilities` function into other scripts for customized workflows.

Dependencies:
- numpy
- pyyaml
- datetime

Example:
    python export_profiles_and_probabilities_yml.py

Author: Joren Bredael
Date: December 7, 2025
"""

import numpy as np
import yaml


def export_wind_profile_shapes_and_probabilities(heights, prl, prp, labels_full, normalisation_wind_speeds, 
                                                wind_directions, n_samples, n_clusters, output_file, 
                                                ref_height=100., n_wind_speed_bins=50, 
                                                wind_direction_bin_width=10., metadata=None):
    """
    Export wind profiles and their probability distributions to YAML file.

    :param heights: Height levels
    :type heights: array
    :param prl: Parallel wind speed components for each cluster
    :type prl: array
    :param prp: Perpendicular wind speed components for each cluster
    :type prp: array
    :param labels_full: Cluster labels for all samples
    :type labels_full: array
    :param normalisation_wind_speeds: Reference wind speeds used for normalization
    :type normalisation_wind_speeds: array
    :param wind_directions: Wind directions in radians (range: -pi to pi)
    :type wind_directions: array
    :param n_samples: Total number of samples
    :type n_samples: int
    :param n_clusters: Number of clusters
    :type n_clusters: int
    :param output_file: Output YAML file path
    :type output_file: str
    :param ref_height: Reference height for scale factor calculation, defaults to 100.
    :type ref_height: float, optional
    :param n_wind_speed_bins: Number of wind speed bins for probability matrix, defaults to 50
    :type n_wind_speed_bins: int, optional
    :param wind_direction_bin_width: Width of wind direction bins in degrees, defaults to 10.
    :type wind_direction_bin_width: float, optional
    :param metadata: Additional metadata dictionary containing location, time range, data source, etc.
    :type metadata: dict, optional
    :return: Probability matrix (3D: n_clusters x n_wind_speed_bins x n_wind_direction_bins)
    :rtype: numpy.ndarray
    """
    
    # Calculate profiles and apply scale factors to ensure magnitude=1 at reference height
    profiles = []
    scale_factors = []
    for i, (u, v) in enumerate(zip(prl, prp)):
        # Calculate magnitude and scale factor at reference height
        w = np.sqrt(u**2 + v**2)
        w_ref = np.interp(ref_height, heights, w)
        
        # Calculate scale factor to make magnitude = 1 at reference height
        if w_ref == 0:
            sf = 1.0
        else:
            sf = 1.0 / w_ref
        
        scale_factors.append(sf)
        
        # Apply scale factor to wind speed components
        u_scaled = u * sf
        v_scaled = v * sf
        
        profile = {
            'id': i+1,
            'height_m': list(map(float, heights)),
            'u_normalized': list(map(float, u_scaled)),
            'v_normalized': list(map(float, v_scaled))
        }
        profiles.append(profile)
    
    # Scale the normalization wind speeds by their corresponding cluster scale factors
    scaled_normalisation_wind_speeds = np.zeros_like(normalisation_wind_speeds)
    for i_cluster in range(n_clusters):
        cluster_mask = labels_full == i_cluster
        scaled_normalisation_wind_speeds[cluster_mask] = normalisation_wind_speeds[cluster_mask] / scale_factors[i_cluster]
    
    # Calculate probability distribution matrix using scaled wind speeds and wind directions
    min_wind_speed = np.min(scaled_normalisation_wind_speeds)
    max_wind_speed = np.max(scaled_normalisation_wind_speeds)
    wind_speed_bins = np.linspace(min_wind_speed, max_wind_speed, n_wind_speed_bins + 1)
    
    # Convert wind directions from radians to degrees (0-360)
    wind_directions_deg = np.degrees(wind_directions)
    # Normalize to 0-360 range
    wind_directions_deg = np.where(wind_directions_deg < 0, wind_directions_deg + 360, wind_directions_deg)
    
    # Create wind direction bins (0 to 360 degrees)
    n_wind_direction_bins = int(360 / wind_direction_bin_width)
    wind_direction_bins = np.linspace(0, 360, n_wind_direction_bins + 1)
    
    # Initialize 3D probability matrix: [n_clusters x n_wind_speed_bins x n_wind_direction_bins]
    probability_matrix = np.zeros((n_clusters, n_wind_speed_bins, n_wind_direction_bins))
    
    for i_cluster in range(n_clusters):
        cluster_mask = labels_full == i_cluster
        cluster_wind_speeds = scaled_normalisation_wind_speeds[cluster_mask]
        cluster_wind_directions = wind_directions_deg[cluster_mask]
        
        # Calculate 2D histogram for this cluster (wind speed vs wind direction)
        hist, _, _ = np.histogram2d(cluster_wind_speeds, cluster_wind_directions, 
                                     bins=[wind_speed_bins, wind_direction_bins])
        
        # Convert to probabilities (percentage of total samples)
        probability_matrix[i_cluster, :, :] = hist / n_samples * 100.0
    
    # Prepare data for YAML export
    base_metadata = {
        'n_clusters': n_clusters,
        'n_wind_speed_bins': n_wind_speed_bins,
        'n_wind_direction_bins': n_wind_direction_bins,
        'wind_direction_bin_width_deg': wind_direction_bin_width,
        'reference_height_m': ref_height,
        'total_samples': n_samples,
        'wind_speed_range_m_s': [float(min_wind_speed), float(max_wind_speed)]
    }

    # Add additional metadata if provided
    if metadata is not None:
        base_metadata.update(metadata)

    # Add timestamp
    from datetime import datetime
    base_metadata['time_created'] = datetime.now().isoformat()

    yml_data = {
        'metadata': base_metadata,
        'altitudes': list(map(float, heights)),  # Export altitude range as a single list
        'wind_speed_bins': {
            'bin_edges_m_s': list(map(float, wind_speed_bins)),
            'bin_centers_m_s': list(map(float, (wind_speed_bins[:-1] + wind_speed_bins[1:]) / 2))
        },
        'wind_direction_bins': {
            'bin_edges_deg': list(map(float, wind_direction_bins)),
            'bin_centers_deg': list(map(float, (wind_direction_bins[:-1] + wind_direction_bins[1:]) / 2))
        },
        'clusters': [
            {
                'id': profile['id'],
                'u_normalized': profile['u_normalized'],
                'v_normalized': profile['v_normalized']
            } for profile in profiles
        ],
        'probability_matrix': {
            'description': 'Probability of each cluster occurring at each wind speed and direction bin (% of total samples)',
            'dimensions': '[n_clusters x n_wind_speed_bins x n_wind_direction_bins]',
            'data': probability_matrix.tolist()
        }
    }
    
    # Export to YAML
    with open(output_file, 'w') as f:
        yaml.dump(yml_data, f, sort_keys=False, default_flow_style=False)
    
    return probability_matrix


if __name__ == '__main__':
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
        from read_data.era5 import read_data
        config = {
            'data_dir': 'data/ERA5',
            'location': {'latitude': 52.0, 'longitude': 4.0},  # Netherlands
            'altitude_range': (0, 500),  # 0-500m above ground
            'years': (2011, 2017)
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
        from read_data.fgw_lidar import read_data
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
        from read_data.dowa import read_data
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
    
    # Import required modules
    from wind_profile_clustering import cluster_normalized_wind_profiles_pca, predict_cluster
    from preprocess_data import preprocess_data

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