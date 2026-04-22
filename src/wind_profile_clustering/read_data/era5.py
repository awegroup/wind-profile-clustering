"""ERA5 Data Reader for Wind Profile Clustering.

This module reads ERA5 reanalysis wind data from NetCDF files and converts it
to the format expected by the wind profile clustering pipeline.

The data consists of:
- Hourly wind components (u, v) at different model levels
- Geopotential height data for altitude calculation (optional)
- Temperature and humidity for altitude calculation (optional)
- Multiple years of data stored in monthly files
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import warnings
import re

# Constants for atmospheric calculations
STANDARD_GRAVITY = 9.80665  # m/s²
R_D = 287.058  # Specific gas constant for dry air [J/(kg·K)]

# Altitude calculation method control
# When True, always use Method 2 (approximate altitudes)
# When False, automatically determine the best method based on available data
FORCE_APPROXIMATE_ALTITUDES = False

# ERA5 L137 Model Level Definitions
# Source: ECMWF https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions 
# a coefficients in Pa, b coefficients dimensionless
L137_COEFFICIENTS = [
    (0, 0.000000, 0.000000),
    (1, 2.000365, 0.000000),
    (2, 3.102241, 0.000000),
    (3, 4.666084, 0.000000),
    (4, 6.827977, 0.000000),
    (5, 9.746966, 0.000000),
    (6, 13.605424, 0.000000),
    (7, 18.608931, 0.000000),
    (8, 24.985718, 0.000000),
    (9, 32.985710, 0.000000),
    (10, 42.879242, 0.000000),
    (11, 54.955463, 0.000000),
    (12, 69.520576, 0.000000),
    (13, 86.895882, 0.000000),
    (14, 107.415741, 0.000000),
    (15, 131.425507, 0.000000),
    (16, 159.279404, 0.000000),
    (17, 191.338562, 0.000000),
    (18, 227.968948, 0.000000),
    (19, 269.539581, 0.000000),
    (20, 316.420746, 0.000000),
    (21, 368.982361, 0.000000),
    (22, 427.592499, 0.000000),
    (23, 492.616028, 0.000000),
    (24, 564.413452, 0.000000),
    (25, 643.339905, 0.000000),
    (26, 729.744141, 0.000000),
    (27, 823.967834, 0.000000),
    (28, 926.344910, 0.000000),
    (29, 1037.201172, 0.000000),
    (30, 1156.853638, 0.000000),
    (31, 1285.610352, 0.000000),
    (32, 1423.770142, 0.000000),
    (33, 1571.622925, 0.000000),
    (34, 1729.448975, 0.000000),
    (35, 1897.519287, 0.000000),
    (36, 2076.095947, 0.000000),
    (37, 2265.431641, 0.000000),
    (38, 2465.770508, 0.000000),
    (39, 2677.348145, 0.000000),
    (40, 2900.391357, 0.000000),
    (41, 3135.119385, 0.000000),
    (42, 3381.743652, 0.000000),
    (43, 3640.468262, 0.000000),
    (44, 3911.490479, 0.000000),
    (45, 4194.930664, 0.000000),
    (46, 4490.817383, 0.000000),
    (47, 4799.149414, 0.000000),
    (48, 5119.895020, 0.000000),
    (49, 5452.990723, 0.000000),
    (50, 5798.344727, 0.000000),
    (51, 6156.074219, 0.000000),
    (52, 6526.946777, 0.000000),
    (53, 6911.870605, 0.000000),
    (54, 7311.869141, 0.000000),
    (55, 7727.412109, 0.000007),
    (56, 8159.354004, 0.000024),
    (57, 8608.525391, 0.000059),
    (58, 9076.400391, 0.000112),
    (59, 9562.682617, 0.000199),
    (60, 10065.978516, 0.000340),
    (61, 10584.631836, 0.000562),
    (62, 11116.662109, 0.000890),
    (63, 11660.067383, 0.001353),
    (64, 12211.547852, 0.001992),
    (65, 12766.873047, 0.002857),
    (66, 13324.668945, 0.003971),
    (67, 13881.331055, 0.005378),
    (68, 14432.139648, 0.007133),
    (69, 14975.615234, 0.009261),
    (70, 15508.256836, 0.011806),
    (71, 16026.115234, 0.014816),
    (72, 16527.322266, 0.018318),
    (73, 17008.789063, 0.022355),
    (74, 17467.613281, 0.026964),
    (75, 17901.621094, 0.032176),
    (76, 18308.433594, 0.038026),
    (77, 18685.718750, 0.044548),
    (78, 19031.289063, 0.051773),
    (79, 19343.511719, 0.059728),
    (80, 19620.042969, 0.068448),
    (81, 19859.390625, 0.077958),
    (82, 20059.931641, 0.088286),
    (83, 20219.664063, 0.099462),
    (84, 20337.863281, 0.111505),
    (85, 20412.308594, 0.124448),
    (86, 20442.078125, 0.138313),
    (87, 20425.718750, 0.153125),
    (88, 20361.816406, 0.168910),
    (89, 20249.511719, 0.185689),
    (90, 20087.085938, 0.203491),
    (91, 19874.025391, 0.222333),
    (92, 19608.572266, 0.242244),
    (93, 19290.226563, 0.263242),
    (94, 18917.460938, 0.285354),
    (95, 18489.707031, 0.308598),
    (96, 18006.925781, 0.332939),
    (97, 17471.839844, 0.358254),
    (98, 16888.687500, 0.384363),
    (99, 16262.046875, 0.411125),
    (100, 15596.695313, 0.438391),
    (101, 14898.453125, 0.466003),
    (102, 14173.324219, 0.493800),
    (103, 13427.769531, 0.521619),
    (104, 12668.257813, 0.549301),
    (105, 11901.339844, 0.576692),
    (106, 11133.304688, 0.603648),
    (107, 10370.175781, 0.630036),
    (108, 9617.515625, 0.655736),
    (109, 8880.453125, 0.680643),
    (110, 8163.375000, 0.704669),
    (111, 7470.343750, 0.727739),
    (112, 6804.421875, 0.749797),
    (113, 6168.531250, 0.770798),
    (114, 5564.382813, 0.790717),
    (115, 4993.796875, 0.809536),
    (116, 4457.375000, 0.827256),
    (117, 3955.960938, 0.843881),
    (118, 3489.234375, 0.859432),
    (119, 3057.265625, 0.873929),
    (120, 2659.140625, 0.887408),
    (121, 2294.242188, 0.899900),
    (122, 1961.500000, 0.911448),
    (123, 1659.476563, 0.922096),
    (124, 1387.546875, 0.931881),
    (125, 1143.250000, 0.940860),
    (126, 926.507813, 0.949064),
    (127, 734.992188, 0.956550),
    (128, 568.062500, 0.963352),
    (129, 424.414063, 0.969513),
    (130, 302.476563, 0.975078),
    (131, 202.484375, 0.980072),
    (132, 122.101563, 0.984542),
    (133, 62.781250, 0.988500),
    (134, 22.835938, 0.991984),
    (135, 3.757813, 0.995003),
    (136, 0.000000, 0.997630),
    (137, 0.000000, 1.000000),
]

def method1_temperature_humidity_altitudes(dsSelected, levels, targetAltitudes, sfcFilePath, location):
    """Calculate altitudes from temperature/humidity data using hydrostatic equation.
    
    Method 1: Uses temperature, humidity, and surface pressure data.
    
    Args:
        dsSelected (xarray.Dataset): Location-selected xarray Dataset.
        levels (array): Model level numbers.
        targetAltitudes (array): Target altitude levels for interpolation [m].
        sfcFilePath (str or Path): Path to surface data file.
        location (dict): Location dictionary with 'latitude' and 'longitude' keys.

    Returns:
        tuple: (windEast, windNorth, altitudes) interpolated to target altitudes.
    """
    print("Using Method 1: Temperature/humidity-based altitude calculation")
    
    # Get required data
    tData = dsSelected['temperature'] if 'temperature' in dsSelected else dsSelected['t']
    qData = dsSelected['specific_humidity'] if 'specific_humidity' in dsSelected else dsSelected['q']
    uData = dsSelected['u_component_of_wind'] if 'u_component_of_wind' in dsSelected else dsSelected['u']
    vData = dsSelected['v_component_of_wind'] if 'v_component_of_wind' in dsSelected else dsSelected['v']
    
    # Get surface pressure
    if not Path(sfcFilePath).exists():
        raise FileNotFoundError(f"Surface data file required for Method 1: {sfcFilePath}")
        
    dsSfc = xr.open_dataset(sfcFilePath)
    dsSfcSel = dsSfc.interp(latitude=location['latitude'], longitude=location['longitude'], method='linear')
    
    # Match the time selection from dsSelected
    # Get the time coordinates that were actually selected
    if hasattr(dsSelected, 'valid_time'):
        timeCoord = 'valid_time'
        selectedTimes = dsSelected.valid_time
    elif hasattr(dsSelected, 'time'):
        timeCoord = 'time' 
        selectedTimes = dsSelected.time
    else:
        raise ValueError("Could not find time coordinate in dsSelected")
        
    # Select matching times in surface data
    dsSfcSel = dsSfcSel.sel(**{timeCoord: selectedTimes})
    sp = dsSfcSel['sp'].values if 'sp' in dsSfcSel else np.exp(dsSfcSel['lnsp'].values)
    dsSfc.close()
    
    # Get surface geopotential from the sfc file (z is included as a static field)
    if 'z' in dsSfcSel:
        zSfc = float(dsSfcSel['z'].values.flat[0])
    elif 'geopotential' in dsSfcSel:
        zSfc = float(dsSfcSel['geopotential'].values.flat[0])
    else:
        raise ValueError("Surface geopotential 'z' not found in surface data file. "
                         "Expected variable 'z' or 'geopotential'.")

    # Calculate altitudes at each timestep using hydrostatic equation
    # This gives altitudes relative to the surface geopotential
    altitudesRelativeToSurface = calculate_geopotential_from_levels(
        tData.values, qData.values, sp, zSfc, L137_COEFFICIENTS, L137_COEFFICIENTS, levels.values
    )
    
    # Convert to altitudes above surface by subtracting the surface altitude
    surfaceAltitude = zSfc / STANDARD_GRAVITY
    altitudesTimeVarying = altitudesRelativeToSurface - surfaceAltitude
    
    # Interpolate wind data to target altitudes at each timestep
    windEastInterp = interpolate_profiles(uData.values, altitudesTimeVarying, targetAltitudes)
    windNorthInterp = interpolate_profiles(vData.values, altitudesTimeVarying, targetAltitudes)
    
    return windEastInterp, windNorthInterp, targetAltitudes


def method2_approximate_altitudes(dsSelected, levels, targetAltitudes):
    """Use approximate altitudes from lookup table and interpolate.
    
    Method 2: Uses predefined model-level-to-altitude mapping based on standard atmosphere.
    
    Args:
        dsSelected (xarray.Dataset): Location-selected xarray Dataset.
        levels (array): Model level numbers.
        targetAltitudes (array): Target altitude levels for interpolation [m].

    Returns:
        tuple: (windEast, windNorth, altitudes) interpolated to target altitudes.
    """
    print("Using Method 2: Approximate altitude calculation")
    
    # Get wind data
    uData = dsSelected['u_component_of_wind'] if 'u_component_of_wind' in dsSelected else dsSelected['u']
    vData = dsSelected['v_component_of_wind'] if 'v_component_of_wind' in dsSelected else dsSelected['v']
    
    # Get approximate altitudes for each level
    levelAltitudeMap = get_pressure_level_altitudes()
    approxAltitudes = np.array([levelAltitudeMap.get(l, np.nan) for l in levels.values])
    
    # Check for missing altitude mappings
    if np.any(np.isnan(approxAltitudes)):
        missingLevels = levels.values[np.isnan(approxAltitudes)]
        raise ValueError(f"No approximate altitudes available for levels: {missingLevels}")
    
    # Create time-varying altitude array (same for all timesteps)
    nTimes = uData.shape[0]
    altitudesTimeVarying = np.tile(approxAltitudes, (nTimes, 1))
    
    # Interpolate wind data to target altitudes at each timestep
    windEastInterp = interpolate_profiles(uData.values, altitudesTimeVarying, targetAltitudes)
    windNorthInterp = interpolate_profiles(vData.values, altitudesTimeVarying, targetAltitudes)
    
    return windEastInterp, windNorthInterp, targetAltitudes


def calculate_geopotential_from_levels(t, q, sp, zSfc, a, b, levels):
    """Calculate geopotential height on model levels using hydrost

atic equation.
    
    Args:
        t (ndarray): Temperature [K] with shape (time, level).
        q (ndarray): Specific humidity [kg/kg] with shape (time, level).
        sp (ndarray): Surface pressure [Pa] with shape (time,).
        zSfc (float or ndarray): Surface geopotential [m²/s²]  (scalar or time).
        a (list): A coefficients [Pa].
        b (list): B coefficients [dimensionless].
        levels (ndarray): Model level numbers (1-based) corresponding to t columns.

    Returns:
        ndarray: Geopotential height [m] with shape (time, level).
    """
    nSamples = t.shape[0]
    nLevelsData = t.shape[1]
    
    # Ensure a and b are numpy arrays
    # a, b are lists of tuples (k, aVal, bVal)
    # We want to access them by index k.
    # Since L137_COEFFICIENTS is sorted by k from 0 to 137, we can just use indices.
    aCoeffs = np.array([x[1] for x in a])
    bCoeffs = np.array([x[2] for x in b])
    
    # Sort data levels from surface (137) to top (1)
    # levels might be [1, 2, ...] or [137, 136, ...]
    # We want to process from bottom up.
    sortIdx = np.argsort(levels)[::-1]  # Descending order (e.g. 137, 136, ...)
    sortedLevels = levels[sortIdx]
    
    # Check if we start at surface
    if sortedLevels[0] != 137:
        warnings.warn(f"Data starts at level {sortedLevels[0]}, not surface (137). Assuming standard atmosphere below.")
        pass

    # Output array for geopotential height (same order as input t)
    h = np.zeros_like(t)
    
    # Initialize geopotential at the lower interface with surface geopotential
    phiLower = np.asarray(zSfc)
    
    # Iterate through sorted levels (bottom to top)
    for i in range(nLevelsData):
        lvl = int(sortedLevels[i])  # e.g. 137
        idx = sortIdx[i]  # index in t
        
        # Layer lvl is between half-level lvl-1 (upper) and lvl (lower)
        # Indices in aCoeffs are same as half-level indices.
        
        # Lower interface pressure (at lvl)
        # phLower = a[lvl] + b[lvl] * sp
        pLower = aCoeffs[lvl] + bCoeffs[lvl] * sp
        
        # Upper interface pressure (at lvl-1)
        # Note: lvl-1 corresponds to half-level index lvl-1
        if lvl == 1:
            # Top layer: upper interface is at level 0 (top of atmosphere)
            pUpper = aCoeffs[0] + bCoeffs[0] * sp  # Should be ~0
        else:
            pUpper = aCoeffs[lvl-1] + bCoeffs[lvl-1] * sp
        
        # Virtual temperature of the layer
        tvLayer = t[:, idx] * (1.0 + 0.609133 * q[:, idx])
        
        # Thickness
        # dPhi = Rd * Tv * ln(pLower / pUpper)
        # Avoid div by zero if pUpper is 0 (top of atmosphere)
        if lvl == 1 or np.any(pUpper <= 0):
            # Top layer or zero pressure: use approximation
            dPhi = R_D * tvLayer * 10.0  # Small layer thickness        
        else:
            dPhi = R_D * tvLayer * np.log(pLower / pUpper)
             
        # Geopotential at upper interface
        phiUpper = phiLower + dPhi
        
        # Geopotential at full level (average)
        phiFull = 0.5 * (phiLower + phiUpper)
        
        # Store result
        h[:, idx] = phiFull / STANDARD_GRAVITY
        
        # Update phiLower for next layer (which is the one above)
        # The upper interface of this layer is the lower interface of the next layer (lvl-1)
        phiLower = phiUpper
        
    return h


def get_pressure_level_altitudes():
    """
    Get accurate altitudes for ERA5 model levels.

    Based on the 1976 version of ICAO standard atmosphere from ECMWF.
    Source: https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions

    :return: Mapping of pressure level to altitude [m]
    :rtype: dict
    """
    # ICAO standard atmosphere altitudes for ERA5 L137 model levels (in meters)
    # Ordered from level 1 (top) to level 137 (surface)
    altitudeLevelsAll = [79301.79, 73721.58, 71115.75, 68618.43, 66210.99, 63890.03, 61651.77, 59492.5, 57408.61,
                          55396.62, 53453.2, 51575.15, 49767.41, 48048.7, 46416.22, 44881.17, 43440.23, 42085, 40808.05,
                          39602.76, 38463.25, 37384.22, 36360.94, 35389.15, 34465, 33585.02, 32746.04, 31945.53, 31177.59,
                          30438.54, 29726.69, 29040.48, 28378.46, 27739.29, 27121.74, 26524.63, 25946.9, 25387.55,
                          24845.63, 24320.28, 23810.67, 23316.04, 22835.68, 22368.91, 21915.16, 21473.98, 21045, 20627.87,
                          20222.24, 19827.95, 19443.55, 19068.35, 18701.27, 18341.27, 17987.41, 17638.78, 17294.53,
                          16953.83, 16616.09, 16281.1, 15948.85, 15619.3, 15292.44, 14968.24, 14646.68, 14327.75, 14011.41,
                          13697.65, 13386.45, 13077.79, 12771.64, 12467.99, 12166.81, 11868.08, 11571.79, 11277.92,
                          10986.7, 10696.22, 10405.61, 10114.89, 9824.08, 9533.2, 9242.26, 8951.3, 8660.32, 8369.35,
                          8078.41, 7787.51, 7496.68, 7205.93, 6915.29, 6624.76, 6334.38, 6044.15, 5754.1, 5464.6, 5176.77,
                          4892.26, 4612.58, 4338.77, 4071.8, 3812.53, 3561.7, 3319.94, 3087.75, 2865.54, 2653.58, 2452.04,
                          2260.99, 2080.41, 1910.19, 1750.14, 1600.04, 1459.58, 1328.43, 1206.21, 1092.54, 987, 889.17,
                          798.62, 714.94, 637.7, 566.49, 500.91, 440.58, 385.14, 334.22, 287.51, 244.68, 205.44, 169.5,
                          136.62, 106.54, 79.04, 53.92, 30.96, 10]
    
    # Create mapping from model level (1-137) to altitude
    levelAltitudeMap = {}
    for i, altitude in enumerate(altitudeLevelsAll):
        level = i + 1  # Model levels are 1-based
        levelAltitudeMap[level] = altitude
    
    return levelAltitudeMap


def interpolate_profiles(data, altitudes, targetAltitudes):
    """Interpolate vertical profiles to target altitudes.
    
    Args:
        data (ndarray): Array of shape (nTime, nLevels).
        altitudes (ndarray): Array of shape (nTime, nLevels).
        targetAltitudes (ndarray): Array of shape (nTargets,).

    Returns:
        ndarray: Interpolated data of shape (nTime, nTargets).
    """
    nTime = data.shape[0]
    nTargets = len(targetAltitudes)
    result = np.zeros((nTime, nTargets))
    
    for t in range(nTime):
        profData = data[t, :]
        profAlt = altitudes[t, :]
        
        # Remove NaNs if any
        valid = ~np.isnan(profAlt) & ~np.isnan(profData)
        if not np.any(valid):
            result[t, :] = np.nan
            continue
            
        profAlt = profAlt[valid]
        profData = profData[valid]
        
        # Sort by altitude
        sortIdx = np.argsort(profAlt)
        profAltSorted = profAlt[sortIdx]
        profDataSorted = profData[sortIdx]
        
        # Interpolate
        result[t, :] = np.interp(targetAltitudes, profAltSorted, profDataSorted)
        
    return result

def read_era5_month(filePath, location, altitudeRange, sfcFilePath=None, targetAltitudes=None):
    """Read ERA5 data for a single month using one of three altitude calculation methods.

    Uses bilinear interpolation to extract data at the specified location, providing 
    more accurate wind data by interpolating between the 4 surrounding ERA5 grid points
    rather than using the nearest neighbor.

    Args:
        filePath (str): Path to ERA5 NetCDF file.
        location (dict): Location specification with 'latitude' and 'longitude' keys.
        altitudeRange (tuple): (minAltitude, maxAltitude) in meters.
        sfcFilePath (str, optional): Path to corresponding surface data file (required for Method 2). Defaults to None.
        targetAltitudes (ndarray, optional): Target altitudes for interpolation [m]. Defaults to None.
    
    Returns:
        dict: Dictionary containing wind data with keys:
            - 'wind_speed_east': East wind component [m/s]
            - 'wind_speed_north': North wind component [m/s]
            - 'datetime': Array of datetime values
            - 'altitude': Array of altitude values [m]
            - 'altitude_method_used': Method used (1 or 2)
            - 'selected_levels': Target altitude levels
    """
    # Input validation
    if location is None:
        raise ValueError("Location must be specified with 'latitude' and 'longitude' keys")
    
    # Open dataset and select location
    ds = xr.open_dataset(filePath)
    
    # Check data format and available variables
    hasTemperature = 'temperature' in ds or 't' in ds
    hasHumidity = 'specific_humidity' in ds or 'q' in ds
    
    # Get model levels
    if 'model_level' in ds:
        levels = ds['model_level']
    elif 'level' in ds:
        levels = ds['level']
    else:
        raise ValueError("No model level coordinate found in dataset")
    
    # Select location using bilinear interpolation
    latTarget = location['latitude']
    lonTarget = location['longitude']
    dsSelected = ds.interp(latitude=latTarget, longitude=lonTarget, method='linear')
    
    # Set up target altitudes
    if targetAltitudes is None:
        minAlt, maxAlt = altitudeRange
        targetAltitudes = np.arange(minAlt, maxAlt + 1, 10)
    
    # Choose and apply altitude calculation method
    if FORCE_APPROXIMATE_ALTITUDES:
        # Method 2: Forced approximate altitudes
        windEast, windNorth, altitudes = method2_approximate_altitudes(
            dsSelected, levels, targetAltitudes
        )
        altitudeMethodUsed = 2
        
    elif hasTemperature and hasHumidity:
        # Method 1: Temperature and humidity available
        if sfcFilePath is None:
            raise ValueError("Surface data file path required for Method 1 (temperature/humidity)")
        
        windEast, windNorth, altitudes = method1_temperature_humidity_altitudes(
            dsSelected, levels, targetAltitudes, sfcFilePath, location
        )
        altitudeMethodUsed = 1
        
    else:
        raise ValueError(
            "Insufficient data for altitude calculation. Need either:\n"
            "- Temperature + humidity + surface pressure + surface geopotential (Method 1), or\n"
            "- Set FORCE_APPROXIMATE_ALTITUDES = True (Method 2)"
        )
    
    # Get time coordinate
    if 'time' in dsSelected:
        datetimeValues = dsSelected.time.values
    elif 'valid_time' in dsSelected:
        datetimeValues = dsSelected.valid_time.values
    else:
        raise ValueError("No time coordinate found in dataset")
    
    result = {
        'wind_speed_east': windEast,
        'wind_speed_north': windNorth,
        'datetime': datetimeValues,
        'altitude': altitudes,
        'altitude_method_used': altitudeMethodUsed,
        'selected_levels': targetAltitudes,
    }
    
    ds.close()
    return result


def read_data(config=None):
    """
    Read ERA5 data for wind profile clustering analysis.

    This function reads multiple ERA5 NetCDF files and combines them into
    a single dataset compatible with the wind profile clustering pipeline.
    
    Uses bilinear interpolation to extract data at specified locations, providing
    more accurate wind data by interpolating between surrounding ERA5 grid points.

    Args:
        config (dict, optional): Configuration dictionary with optional keys:
            - 'data_dir': Path to data directory (default: relative to script)
            - 'location': Dict with 'latitude' and 'longitude' keys
            - 'altitude_range': Tuple (minAlt, maxAlt) in meters
            - 'years': Tuple (startYear, endYear) to limit data range
    
    Returns:
        dict: Dictionary containing:
            - 'wind_speed_east': East wind component [m/s] (nSamples, nAltitudes)
            - 'wind_speed_north': North wind component [m/s] (nSamples, nAltitudes)
            - 'n_samples': Number of time samples
            - 'datetime': Array of datetime values
            - 'altitude': Array of altitude values [m]
            - 'years': Tuple of (firstYear, lastYear)
    """
    # Suppress xarray backend loading warnings (cfgrib is optional)
    warnings.filterwarnings('ignore', message='.*cfgrib.*', category=RuntimeWarning)
    
    # Default configuration
    if config is None:
        config = {}

    defaultDataDir = Path(__file__).parent.parent.parent.parent / 'data' / 'era5'
    dataDir = Path(config.get('data_dir', defaultDataDir))
    location = config.get('location', None)
    altitudeRange = config.get('altitude_range', (0, 1000))
    years = config.get('years', None)

    # Only accept ml_*.netcdf
    filePattern = 'ml_*.netcdf'

    # Find all ERA5 wind data files matching pattern
    windFiles = sorted(glob(str(dataDir / filePattern)))

    if not windFiles:
        raise FileNotFoundError(f"No ERA5 files found matching pattern {filePattern} in {dataDir}")

    print(f"Found {len(windFiles)} ERA5 data files")

    # Filter by years if specified
    if years is not None:
        startYear, endYear = years
        filteredFiles = []
        for filePath in windFiles:
            fileName = Path(filePath).name
            # Robustly extract year using regex (matches 4 consecutive digits starting with 19 or 20)
            # Expecting ml_YYYY_MM.netcdf
            match = re.search(r'ml_(\d{4})_\d{2}\.netcdf', fileName)
            if match:
                year = int(match.group(1))
                if startYear <= year <= endYear:
                    filteredFiles.append(filePath)
            else:
                warnings.warn(f"Could not extract year from filename: {fileName}")
        windFiles = filteredFiles
        print(f"Filtered to {len(windFiles)} files for years {startYear}-{endYear}")

    if not windFiles:
        raise ValueError("No files remain after year filtering")

    # Define target altitudes for interpolation
    minAlt, maxAlt = altitudeRange
    targetAltitudes = np.arange(minAlt, maxAlt + 1, 10) # 10m resolution

    # Read all monthly files
    allData = []
    for i, filePath in enumerate(windFiles):
        print(f"Reading file {i+1}/{len(windFiles)}: {Path(filePath).name}")
        
        # Find corresponding sfc file
        # Expecting sfc_YYYY_MM.netcdf matches ml_YYYY_MM.netcdf
        fileName = Path(filePath).name
        sfcName = fileName.replace('ml_', 'sfc_')
        sfcFilePath = Path(filePath).parent / sfcName
        
        if not sfcFilePath.exists():
             # Try looking in parent/ERA5 if current is ERA5paper
             if Path(filePath).parent.name == 'ERA5paper':
                 sfcFilePath = Path(filePath).parent.parent / 'ERA5' / sfcName
        
        try:
            monthData = read_era5_month(filePath, location, altitudeRange, sfcFilePath, targetAltitudes=targetAltitudes)
            allData.append(monthData)
        except Exception as e:
            warnings.warn(f"Error reading {filePath}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not allData:
        raise ValueError("No data could be read from any files")

    # Combine all monthly data
    combinedWindEast = np.concatenate([data['wind_speed_east'] for data in allData], axis=0)
    combinedWindNorth = np.concatenate([data['wind_speed_north'] for data in allData], axis=0)
    combinedDatetime = np.concatenate([data['datetime'] for data in allData])

    # Use altitude from first file (should be consistent)
    altitude = allData[0]['altitude']

    # Calculate year range
    yearsArray = pd.to_datetime(combinedDatetime).year
    yearRange = (yearsArray.min(), yearsArray.max())

    nSamples = len(combinedDatetime)
    # Report which altitude methods were used
    methodsUsed = [data.get('altitude_method_used') for data in allData if data.get('altitude_method_used') is not None]
    uniqueMethods = set()
    
    if methodsUsed:
        uniqueMethods = set(methodsUsed)
        methodNames = {1: "Temperature/Humidity", 2: "Approximate"}
        methodsStr = ", ".join([f"Method {m} ({methodNames.get(m, 'Unknown')})" for m in sorted(uniqueMethods)])
        print(f"Altitude calculation methods used: {methodsStr}")
    else:
        uniqueMethods = {2}  # Default to Method 2 if no methods recorded
    
    print(f"Total samples: {nSamples}")
    print(f"Altitude range: {altitude.min():.1f} - {altitude.max():.1f} m")
    print(f"Time range: {combinedDatetime[0]} to {combinedDatetime[-1]}")
    print(f"Years: {yearRange[0]} - {yearRange[1]}")

    return {
        'wind_speed_east': combinedWindEast,
        'wind_speed_north': combinedWindNorth,
        'n_samples': nSamples,
        'datetime': combinedDatetime,
        'altitude': altitude,
        'years': yearRange,
    }



if __name__ == '__main__':
    # Example usage
    print("Reading ERA5 data...")
    
    # Read data for a specific location and altitude range
    config = {
        'location': {'latitude': 52.0, 'longitude': 4.0},  # Netherlands
        'altitude_range': (10, 500),  
        'years': (2010, 2017)  
    }
    
    try:
        data = read_data(config)
        
        print(f"\nData summary:")
        print(f"  Shape: {data['wind_speed_east'].shape}")
        print(f"  Altitudes: {data['altitude']}")
        print(f"  Sample wind speeds (first timestep):")
        print(f"    East: {data['wind_speed_east'][0, :]}")
        print(f"    North: {data['wind_speed_north'][0, :]}")
        
    except Exception as e:
        print(f"Error reading data: {e}")
