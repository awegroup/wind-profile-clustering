"""Run wind profile fitting analysis and export results to YAML.

This script loads wind data from various sources (ERA5, FGW lidar, or DOWA),
fits a logarithmic or power law wind speed profile to it, and exports the
result to YAML format for further use.

Wind speed is computed as the horizontal magnitude sqrt(u_east**2 + u_north**2).
The exported profile has u_normalized equal to the fitted shape (normalised
to 1 at the reference height) and v_normalized set to zero.
"""

import sys
from pathlib import Path
import numpy as np

# Add src directory to path for imports
srcPath = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(srcPath))

from wind_profile_clustering.fitting_and_prescribing.fit_profile import fit_wind_profile
from wind_profile_clustering.export_profiles_and_probabilities_yml import (
    export_wind_profile_shapes_and_probabilities
)


def main():
    """Run wind profile fitting analysis and export results.

    User can configure:
    - DATA_SOURCE: Choose between 'era5', 'fgw_lidar', or 'dowa'
    - PROFILE_TYPE: 'logarithmic' or 'power_law'
    - REF_HEIGHT: Reference height for profile normalisation (in metres)
    - AWESIO_VALIDATE: Validate the exported YAML with awesio (default: True)
    """
    # =============================================================================
    # USER CONFIGURATION
    # =============================================================================
    PROFILE_TYPE = 'logarithmic'  # 'logarithmic' or 'power_law'
    REF_HEIGHT = 200.0            # Reference height for normalisation [m]
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
            'location': {'latitude': 52.0, 'longitude': 4.0},  # Netherlands
            'altitude_range': (10, 500),  # 10-500 m above ground
            'years': (2024, 2025)
        }
        data = read_data(config)
        outPrefix = 'era5'

        locationMeta = {
            'latitude': config['location']['latitude'],
            'longitude': config['location']['longitude']
        }
        timeRangeMeta = {
            'start_year': config['years'][0],
            'end_year': config['years'][1],
            'years_included': list(range(config['years'][0], config['years'][1] + 1)),
            'months_included': 'all'
        }
        dataSourceLabel = 'ERA5'
        altitudeRangeMeta = list(config['altitude_range'])

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
            'start_year': data['years'][0],
            'end_year': data['years'][1],
            'years_included': [],
            'months_included': 'varies'
        }
        dataSourceLabel = 'FGW_Lidar'
        altitudeRangeMeta = [float(data['altitude'].min()), float(data['altitude'].max())]

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
            'start_year': data['years'][0],
            'end_year': data['years'][1],
            'years_included': [],
            'months_included': 'varies'
        }
        dataSourceLabel = 'DOWA'
        altitudeRangeMeta = [float(data['altitude'].min()), float(data['altitude'].max())]

    elif DATA_SOURCE == 'wls7_lidar':
        print("Using WindCube WLS7-130 lidar data...")
        from wind_profile_clustering.read_data.wls7_130_lidar import read_data
        config = {
            'data_dir': 'data/WLS7-130_lidar',
            'date_range': None,      # e.g. ('2024-10-20', '2025-06-11')
            'resample_hourly': True, # resample ~4-second data to hourly means
        }
        data = read_data(config)
        outPrefix = 'wls7_lidar'

        locationMeta = {'latitude': 54.1254, 'longitude': -9.7801}
        timeRangeMeta = {
            'start_year': data['years'][0],
            'end_year': data['years'][1],
            'years_included': [],
            'months_included': 'varies',
        }
        dataSourceLabel = 'WLS7-130_Lidar'
        altitudeRangeMeta = [float(data['altitude'].min()), float(data['altitude'].max())]

    else:
        raise ValueError(
            f"Unknown data source: {DATA_SOURCE}. "
            "Choose from 'era5', 'fgw_lidar', 'dowa', or 'wls7_lidar'."
        )

    print(f"Loaded {data['n_samples']} samples from {data['years'][0]} to {data['years'][1]}")
    print(f"Altitude range: {data['altitude'].min():.1f} - {data['altitude'].max():.1f} m")

    # Fit wind profile
    print(f"\nFitting {PROFILE_TYPE} wind profile...")
    fitResults = fit_wind_profile(data, profileType=PROFILE_TYPE, refHeight=REF_HEIGHT)
    print(f"Fit parameters: {fitResults['fitParams']}")

    # Build per-profile-type label and note describing what happened to wind components
    profileLabels = {
        'logarithmic': 'logarithmic  U(z) = (u*/kappa) * ln(z/z0)',
        'power_law':   'power law  U(z) = U_ref * (z/z_ref)**alpha',
    }
    profileLabel = profileLabels[PROFILE_TYPE]
    note = (
        f"Wind speed magnitude sqrt(u_east**2 + u_north**2) was computed at each altitude and "
        f"timestep. A {profileLabel} profile was fitted to the time-averaged wind speed "
        f"profile. u_normalized contains the fitted profile normalised to 1 at "
        f"{REF_HEIGHT:.0f} m; v_normalized is zero for all altitudes. "
        f"Fit parameters: {fitResults['fitParams']}."
    )

    # Prepare metadata
    nameLabel = PROFILE_TYPE.replace('_', ' ').title()
    metadata = {
        'name': f'{dataSourceLabel} Wind Profile {nameLabel} Fit',
        'description': (
            f'Wind profile obtained by fitting a {PROFILE_TYPE} profile '
            f'to {dataSourceLabel} data'
        ),
        'note': note,
        'data_source': dataSourceLabel,
        'location': locationMeta,
        'time_range': timeRangeMeta,
        'altitude_range': altitudeRangeMeta,
    }

    # Export to YAML
    print("\nExporting results to YAML...")
    outputFile = f'results/wind_profile_fit_{PROFILE_TYPE}_{outPrefix}.yml'

    probMatrix = export_wind_profile_shapes_and_probabilities(
        data['altitude'],
        fitResults['prl'],
        fitResults['prp'],
        fitResults['labelsFull'],
        fitResults['normalisationWindSpeeds'],
        fitResults['windDirections'],
        fitResults['nSamples'],
        1,  # single fitted profile
        outputFile,
        metadata=metadata,
        refHeight=REF_HEIGHT,
        validate=AWESIO_VALIDATE,
    )

    print(f"\nExported fitted profile to: {outputFile}")
    print(f"Probability matrix shape: {probMatrix.shape}")
    print(f"Total probability sum: {np.sum(probMatrix):.2f}%")


if __name__ == '__main__':
    main()
