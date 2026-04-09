"""Export wind profile shapes and probability distributions to YAML.

- Metadata about the clustering process and data source.
- A single list of altitude levels shared across all clusters.
- Normalized wind profile components (u and v) for each cluster.
- Probability distributions of wind speeds for each cluster.
"""

import importlib.metadata

import numpy as np
import yaml
from datetime import datetime


def _get_awesio_version():
    """Return the installed awesio package version, or 'unknown' if not available."""
    try:
        return importlib.metadata.version('awesio')
    except importlib.metadata.PackageNotFoundError:
        print("Warning: awesio package not found. Version will be set to 'unknown' in metadata.")
        return 'unknown'


def export_wind_profile_shapes_and_probabilities(heights, prl, prp, labelsFull, normalisationWindSpeeds,
                                                windDirections, nSamples, nClusters, outputFile,
                                                refHeight=100., nWindSpeedBins=50,
                                                windDirectionBinWidth=36., metadata=None,
                                                validate=False):
    """Export wind profiles and their probability distributions to YAML file.

    Args:
        heights (array): Height levels.
        prl (array): Parallel wind speed components for each cluster.
        prp (array): Perpendicular wind speed components for each cluster.
        labelsFull (array): Cluster labels for all samples.
        normalisationWindSpeeds (array): Reference wind speeds used for normalization.
        windDirections (array): Wind directions in radians (range: -pi to pi).
        nSamples (int): Total number of samples.
        nClusters (int): Number of clusters.
        outputFile (str): Output YAML file path.
        refHeight (float): Reference height for scale factor calculation. Defaults to 100.0.
        nWindSpeedBins (int): Number of wind speed bins for probability matrix. Defaults to 50.
        windDirectionBinWidth (float): Width of wind direction bins in degrees. Defaults to 36.0.
        metadata (dict): Metadata dictionary containing location, time range, data source, etc.
            Must include 'name', 'description', 'note', and 'data_source'. 'awesIO_version' is
            always determined from the installed package and must not be included.
        validate (bool): Validate the exported YAML using awesio.validator. Defaults to False.

    Returns:
        np.ndarray: Probability matrix with shape (nClusters, nWindSpeedBins, nWindDirectionBins).
    """
    # Calculate profiles and apply scale factors to ensure magnitude=1 at reference height
    profiles = []
    scaleFactors = []
    for i, (u, v) in enumerate(zip(prl, prp)):
        # Interpolate u and v components at reference height to get exact values
        uRef = np.interp(refHeight, heights, u)
        vRef = np.interp(refHeight, heights, v)
        
        # Calculate magnitude at reference height from interpolated components
        wRef = np.sqrt(uRef**2 + vRef**2)

        # Calculate scale factor to make magnitude = 1 at reference height
        if wRef == 0:
            sf = 1.0
        else:
            sf = 1.0 / wRef

        scaleFactors.append(sf)

        # Apply scale factor to wind speed components
        uScaled = u * sf
        vScaled = v * sf

        profile = {
            'id': i+1,
            'u_normalized': list(map(float, uScaled)),
            'v_normalized': list(map(float, vScaled))
        }
        profiles.append(profile)

    # Scale the normalization wind speeds by their corresponding cluster scale factors
    scaledNormalisationWindSpeeds = np.zeros_like(normalisationWindSpeeds)
    for iCluster in range(nClusters):
        clusterMask = labelsFull == iCluster
        scaledNormalisationWindSpeeds[clusterMask] = normalisationWindSpeeds[clusterMask] / scaleFactors[iCluster]

    # Calculate probability distribution matrix using scaled wind speeds and wind directions
    minWindSpeed = np.min(scaledNormalisationWindSpeeds)
    maxWindSpeed = np.max(scaledNormalisationWindSpeeds)
    windSpeedBins = np.linspace(minWindSpeed, maxWindSpeed, nWindSpeedBins + 1)

    # Convert wind directions from radians to degrees (0-360)
    windDirectionsDeg = np.degrees(windDirections)
    # Normalize to 0-360 range
    windDirectionsDeg = np.where(windDirectionsDeg < 0, windDirectionsDeg + 360, windDirectionsDeg)

    # Create wind direction bins (0 to 360 degrees)
    nWindDirectionBins = int(360 / windDirectionBinWidth)
    windDirectionBins = np.linspace(0, 360, nWindDirectionBins + 1)

    # Initialize 3D probability matrix: [nClusters x nWindSpeedBins x nWindDirectionBins]
    probabilityMatrix = np.zeros((nClusters, nWindSpeedBins, nWindDirectionBins))

    for iCluster in range(nClusters):
        clusterMask = labelsFull == iCluster
        clusterWindSpeeds = scaledNormalisationWindSpeeds[clusterMask]
        clusterWindDirections = windDirectionsDeg[clusterMask]

        # Calculate 2D histogram for this cluster (wind speed vs wind direction)
        hist, _, _ = np.histogram2d(clusterWindSpeeds, clusterWindDirections,
                                     bins=[windSpeedBins, windDirectionBins])

        # Convert to probabilities (percentage of total samples)
        probabilityMatrix[iCluster, :, :] = hist / nSamples * 100.0

    # Validate and merge caller-supplied metadata
    if metadata is None:
        raise ValueError("metadata must be provided")
    for required_key in ('name', 'description', 'note', 'data_source'):
        if required_key not in metadata:
            raise ValueError(f"metadata is missing required key: '{required_key}'")


    # Prepare data for YAML export — start with computed fields
    metadatadict = {
        'name': metadata['name'],
        'description': metadata['description'],
        'note': metadata['note'],
        'awesIO_version': _get_awesio_version(),
        'schema': 'wind_resource_schema.yml',
        'data_source': metadata['data_source'],
        'time_created': datetime.now().isoformat(),
        'location': metadata.get('location',{}),
        'time_range': metadata.get('time_range',{}),
        'altitude_range': metadata.get('altitude_range', []),
        'n_clusters': nClusters,
        'n_wind_speed_bins': nWindSpeedBins,
        'n_wind_direction_bins': nWindDirectionBins,
        'wind_direction_bin_width': windDirectionBinWidth,
        'reference_height': refHeight,
        'total_samples': nSamples,
        'wind_speed_range': [float(minWindSpeed), float(maxWindSpeed)]
    }



    ymlData = {
        'metadata': metadatadict,
        'altitudes': list(map(float, heights)),
        'wind_speed_bins': {
            'bin_edges': list(map(float, windSpeedBins)),
            'bin_centers': list(map(float, (windSpeedBins[:-1] + windSpeedBins[1:]) / 2))
        },
        'wind_direction_bins': {
            'bin_edges': list(map(float, windDirectionBins)),
            'bin_centers': list(map(float, (windDirectionBins[:-1] + windDirectionBins[1:]) / 2))
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
            'data': probabilityMatrix.tolist()
        }
    }

    # Export to YAML
    with open(outputFile, 'w') as f:
        yaml.dump(ymlData, f, sort_keys=False, default_flow_style=False)

    # Validate exported YAML using awesio validator if requested
    if validate:
        from awesio.validator import validate as awesio_validate
        awesio_validate(outputFile)
        print("Schema validation passed.")

    return probabilityMatrix