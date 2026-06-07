"""Load wind data, perform clustering, and fit power law and logarithmic profiles.

This script loads wind data from various sources (ERA5, FGW lidar, or DOWA),
performs clustering using PCA and K-means, fits a power law profile and a
logarithmic profile to the data, and exports all three results to separate
YAML files for further use.
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
from wind_profile_clustering.fitting_and_prescribing.fit_profile import fit_wind_profile
from wind_profile_clustering.export_profiles_and_probabilities_yml import (
    export_wind_profile_shapes_and_probabilities
)


def main():
    """Load data, cluster wind profiles, and fit power law and logarithmic profiles.

    User can configure:
    - DATA_SOURCE: Choose between 'era5', 'fgw_lidar', 'dowa', or 'wls7_lidar'
    - N_CLUSTERS: Number of clusters to create (default: 8)
    - REF_HEIGHT: Reference height for profile normalisation (in metres)
    - SAVE_PLOTS: Save clustering plots as PDF files (default: True)
    - AWESIO_VALIDATE: Validate exported YAML with awesio (default: True)
    """
    # =============================================================================
    # USER CONFIGURATION
    # =============================================================================
    N_CLUSTERS = 8       # Number of clusters to create
    SAVE_PLOTS = True    # Save clustering plots as PDF files in results/ directory
    REF_HEIGHT = 200.0   # Reference height for profile normalisation [m]
    AWESIO_VALIDATE = True

    # =============================================================================
    # DATA SOURCE CONFIGURATION
    # =============================================================================
    # Choose which data source to use by setting DATA_SOURCE to one of:
    # 'era5'       - Use ERA5 reanalysis data from NetCDF files
    # 'fgw_lidar'  - Use FGW lidar measurements from CSV file
    # 'dowa'       - Use DOWA model data from NetCDF files
    # 'wls7_lidar' - Use WindCube WLS7-130 lidar data from RTD files
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
            'location': {'latitude': 54.12645912483461, 'longitude': -9.781319131175586},  # Bangor Erris, Ireland
            # 'location': {'latitude': 51.970282335517595, 'longitude': 3.973682107506852},  # Tweede Maasvlakte, Netherlands
            'altitude_range': (10, 500),  # 10-500 m above ground
            'years': (2010, 2020)
        }
        data = read_data(config)
        outPrefix = 'era5'

        locationMeta = {
            'latitude': config['location']['latitude'],
            'longitude': config['location']['longitude']
        }
        timeRangeMeta = {
            'start_date': str(data['datetime'][0].astype('datetime64[D]')),
            'end_date': str(data['datetime'][-1].astype('datetime64[D]')),
        }
        altitudeRangeMeta = list(config['altitude_range'])
        dataSourceLabel = 'ERA5'

    elif DATA_SOURCE == 'fgw_lidar':
        print("Using FGW lidar data...")
        from wind_profile_clustering.read_data.fgw_lidar import read_data
        config = {
            'data_dir': 'data/fgw_lidar',
            'read_raw_data': False
        }
        data = read_data(config)
        outPrefix = 'fgw_lidar'

        locationMeta = {'latitude': None, 'longitude': None}
        timeRangeMeta = {
            'start_date': str(data['datetime'][0].astype('datetime64[D]')),
            'end_date': str(data['datetime'][-1].astype('datetime64[D]')),
        }
        altitudeRangeMeta = [float(data['altitude'].min()), float(data['altitude'].max())]
        dataSourceLabel = 'FGW_Lidar'

    elif DATA_SOURCE == 'dowa':
        print("Using DOWA model data...")
        from wind_profile_clustering.read_data.dowa import read_data
        config = {
            'data_dir': 'data/dowa',
            'name': 'mmij'
        }
        data = read_data(config)
        outPrefix = 'dowa_mmij'

        locationMeta = {'latitude': None, 'longitude': None}
        timeRangeMeta = {
            'start_date': str(data['datetime'][0].astype('datetime64[D]')),
            'end_date': str(data['datetime'][-1].astype('datetime64[D]')),
        }
        altitudeRangeMeta = [float(data['altitude'].min()), float(data['altitude'].max())]
        dataSourceLabel = 'DOWA'

    elif DATA_SOURCE == 'wls7_lidar':
        print("Using WindCube WLS7-130 lidar data...")
        from wind_profile_clustering.read_data.wls7_130_lidar import read_data
        config = {
            'data_dir': 'data/WLS7-130_lidar',
            'date_range': None,
            'resample_hourly': True,
        }
        data = read_data(config)
        outPrefix = 'wls7_lidar'

        locationMeta = {'latitude': 54.1254, 'longitude': -9.7801}
        timeRangeMeta = {
            'start_date': str(data['datetime'][0].astype('datetime64[D]')),
            'end_date': str(data['datetime'][-1].astype('datetime64[D]')),
        }
        altitudeRangeMeta = [float(data['altitude'].min()), float(data['altitude'].max())]
        dataSourceLabel = 'WLS7-130_Lidar'

    else:
        raise ValueError(
            f"Unknown data source: {DATA_SOURCE}. "
            "Choose from 'era5', 'fgw_lidar', 'dowa', or 'wls7_lidar'."
        )

    print(f"Loaded {data['n_samples']} samples from {data['years'][0]} to {data['years'][1]}")
    print(f"Altitude range: {data['altitude'].min():.1f} - {data['altitude'].max():.1f} m")

    # =========================================================================
    # STEP 1: CLUSTERING
    # =========================================================================
    print(f"\nPerforming clustering with {N_CLUSTERS} clusters...")
    results = perform_clustering_analysis(data, N_CLUSTERS, ref_height=REF_HEIGHT)

    processedData = results['processedData']
    processedDataFull = results['processedDataFull']
    res = results['clusteringResults']
    labelsFull = results['labelsFull']
    frequencyClusters = results['frequencyClusters']

    print("\nGenerating clustering visualizations...")
    plot_all_results(
        processedData, res, processedDataFull, labelsFull,
        frequencyClusters, N_CLUSTERS, savePlots=SAVE_PLOTS
    )

    clusteringMetadata = {
        'name': f'{dataSourceLabel} Wind Profile Clustering',
        'description': (
            f'Wind profile clustering results derived from {dataSourceLabel} data '
            f'using PCA and K-means with {N_CLUSTERS} clusters'
        ),
        'note': (
            f'Clustering performed on normalised wind profiles. '
            f'Low wind speed samples (mean < 5 m/s) were excluded from training.'
        ),
        'data_source': dataSourceLabel,
        'location': locationMeta,
        'time_range': timeRangeMeta,
        'altitude_range': altitudeRangeMeta,
    }

    print("\nExporting clustering results to YAML...")
    clusteringOutputFile = f'results/wind_profiles_and_probabilities_{outPrefix}.yml'
    prl = res['clusters_feature']['parallel']
    prp = res['clusters_feature']['perpendicular']
    clusteringProbMatrix = export_wind_profile_shapes_and_probabilities(
        data['altitude'],
        prl,
        prp,
        labelsFull,
        processedDataFull['normalisation_value'],
        processedDataFull['reference_vector_direction'],
        processedDataFull['n_samples'],
        N_CLUSTERS,
        clusteringOutputFile,
        refHeight=REF_HEIGHT,
        metadata=clusteringMetadata,
        validate=AWESIO_VALIDATE,
    )
    print(f"Exported clustering results to: {clusteringOutputFile}")
    print(f"Probability matrix shape: {clusteringProbMatrix.shape}")
    print(f"Total probability sum: {np.sum(clusteringProbMatrix):.2f}%")

    # =========================================================================
    # STEP 2: POWER LAW FIT
    # =========================================================================
    print("\nFitting power law wind profile...")
    fitResultsPower = fit_wind_profile(data, profileType='power_law', refHeight=REF_HEIGHT)
    print(f"Power law fit parameters: {fitResultsPower['fitParams']}")

    powerLawNote = (
        f"Wind speed magnitude sqrt(u_east**2 + u_north**2) was computed at each altitude and "
        f"timestep. A power law profile U(z) = U_ref * (z/z_ref)**alpha was fitted to the "
        f"time-averaged wind speed profile. u_normalized contains the fitted profile normalised "
        f"to 1 at {REF_HEIGHT:.0f} m; v_normalized is zero for all altitudes. "
        f"Fit parameters: {fitResultsPower['fitParams']}."
    )
    powerLawMetadata = {
        'name': f'{dataSourceLabel} Wind Profile Power Law Fit',
        'description': (
            f'Wind profile obtained by fitting a power law profile to {dataSourceLabel} data'
        ),
        'note': powerLawNote,
        'data_source': dataSourceLabel,
        'location': locationMeta,
        'time_range': timeRangeMeta,
        'altitude_range': altitudeRangeMeta,
    }

    print("\nExporting power law fit results to YAML...")
    powerLawOutputFile = f'results/wind_profile_fit_power_law_{outPrefix}.yml'
    powerLawProbMatrix = export_wind_profile_shapes_and_probabilities(
        data['altitude'],
        fitResultsPower['prl'],
        fitResultsPower['prp'],
        fitResultsPower['labelsFull'],
        fitResultsPower['normalisationWindSpeeds'],
        fitResultsPower['windDirections'],
        fitResultsPower['nSamples'],
        1,
        powerLawOutputFile,
        refHeight=REF_HEIGHT,
        metadata=powerLawMetadata,
        validate=AWESIO_VALIDATE,
    )
    print(f"Exported power law fit to: {powerLawOutputFile}")
    print(f"Total probability sum: {np.sum(powerLawProbMatrix):.2f}%")

    # =========================================================================
    # STEP 3: LOGARITHMIC FIT
    # =========================================================================
    print("\nFitting logarithmic wind profile...")
    fitResultsLog = fit_wind_profile(data, profileType='logarithmic', refHeight=REF_HEIGHT)
    print(f"Logarithmic fit parameters: {fitResultsLog['fitParams']}")

    logNote = (
        f"Wind speed magnitude sqrt(u_east**2 + u_north**2) was computed at each altitude and "
        f"timestep. A logarithmic profile U(z) = (u*/kappa) * ln(z/z0) was fitted to the "
        f"time-averaged wind speed profile. u_normalized contains the fitted profile normalised "
        f"to 1 at {REF_HEIGHT:.0f} m; v_normalized is zero for all altitudes. "
        f"Fit parameters: {fitResultsLog['fitParams']}."
    )
    logMetadata = {
        'name': f'{dataSourceLabel} Wind Profile Logarithmic Fit',
        'description': (
            f'Wind profile obtained by fitting a logarithmic profile to {dataSourceLabel} data'
        ),
        'note': logNote,
        'data_source': dataSourceLabel,
        'location': locationMeta,
        'time_range': timeRangeMeta,
        'altitude_range': altitudeRangeMeta,
    }

    print("\nExporting logarithmic fit results to YAML...")
    logOutputFile = f'results/wind_profile_fit_logarithmic_{outPrefix}.yml'
    logProbMatrix = export_wind_profile_shapes_and_probabilities(
        data['altitude'],
        fitResultsLog['prl'],
        fitResultsLog['prp'],
        fitResultsLog['labelsFull'],
        fitResultsLog['normalisationWindSpeeds'],
        fitResultsLog['windDirections'],
        fitResultsLog['nSamples'],
        1,
        logOutputFile,
        refHeight=REF_HEIGHT,
        metadata=logMetadata,
        validate=AWESIO_VALIDATE,
    )
    print(f"Exported logarithmic fit to: {logOutputFile}")
    print(f"Total probability sum: {np.sum(logProbMatrix):.2f}%")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n=== Export summary ===")
    print(f"  Clustering ({N_CLUSTERS} clusters): {clusteringOutputFile}")
    print(f"  Power law fit:                    {powerLawOutputFile}")
    print(f"  Logarithmic fit:                  {logOutputFile}")

    plt.show()


if __name__ == '__main__':
    main()
