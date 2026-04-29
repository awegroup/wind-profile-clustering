"""Lidar Data Reader for Wind Profile Analysis.

This module reads WindCube WLS7-130 lidar data from .rtd files and converts
them to the format expected by the wind profile clustering pipeline.
"""

import warnings
import re
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd


HEADER_SIZE_KEY = "HeaderSize="


def parse_rtd_header(file_path):
    """Parse the header section of an RTD file to extract instrument metadata.

    Args:
        file_path (str or Path): Path to the .rtd file.

    Returns:
        dict: Metadata including:
            - 'n_header_lines' (int): Number of header lines before the column row.
            - 'altitudes' (ndarray): Measurement altitude gates [m].
            - 'location_name' (str): Site name.
    """
    with open(file_path, 'r', encoding='latin-1') as f:
        firstLine = f.readline().strip()

    nHeaderLines = 40  # WLS7-130 default
    if firstLine.startswith(HEADER_SIZE_KEY):
        nHeaderLines = int(firstLine[len(HEADER_SIZE_KEY):])

    metadata = {'n_header_lines': nHeaderLines}

    with open(file_path, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            if i >= nHeaderLines:
                break
            line = line.strip()
            if line.startswith('Altitudes (m)='):
                altStr = line.split('=', 1)[1].strip()
                metadata['altitudes'] = np.array([float(a) for a in altStr.split()])
            elif line.startswith('Location='):
                metadata['location_name'] = line.split('=', 1)[1].strip()
            elif line.startswith('GPS Location='):
                metadata['gps_location'] = line.split('=', 1)[1].strip()

    return metadata


def read_rtd_file(file_path):
    """Read a single RTD lidar file and extract horizontal wind profile data.

    Only rows with Position == '270' are retained.  These rows contain the
    synthesised horizontal wind vector after a complete 4-beam DBS sweep.

    Args:
        file_path (str or Path): Path to the .rtd file.

    Returns:
        tuple: (pandas.DataFrame indexed by datetime, metadata dict).
            Returns (None, None) on failure.
    """
    file_path = Path(file_path)
    metadata = parse_rtd_header(file_path)

    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            skiprows=metadata['n_header_lines'] + 1,  # +1 to skip the '***' separator line
            header=0,
            index_col=False,
            na_values=['NaN', 'nan', ''],
            low_memory=False,
            engine='c',
            encoding='latin-1',
            on_bad_lines='skip',
        )
    except Exception as e:
        warnings.warn(f"Failed to read {file_path.name}: {e}")
        return None, None

    df.columns = [c.strip() for c in df.columns]

    if 'Position' not in df.columns:
        warnings.warn(f"No 'Position' column in {file_path.name}")
        return None, None

    # Keep only complete-scan rows (Position == 270) which carry the wind vector.
    # Position is object dtype (string) because beam 'V' prevents numeric coercion.
    df = df[df['Position'].astype(str).str.strip() == '270'].copy()

    if df.empty:
        return None, None

    df['datetime'] = pd.to_datetime(
        df['Timestamp'].str.strip(),
        format='%Y/%m/%d %H:%M:%S.%f',
        errors='coerce',
    )
    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime').sort_index()

    return df, metadata


def extract_wind_components(df, altitudes):
    """Extract east (X) and north (Y) wind components at each altitude gate.

    Args:
        df (pandas.DataFrame): RTD data filtered to Position == 270 rows.
        altitudes (ndarray): Altitude gates to extract [m].

    Returns:
        tuple: (windEast, windNorth, validAltitudes) arrays of shape
            (nTime, nAltitudes).
    """
    eastCols, northCols, validAlt = [], [], []

    for alt in altitudes:
        # In the RTD file: X-wind is the northward FROM component and Y-wind is the
        # eastward FROM component (positive when wind comes FROM that direction).
        # The "Wind Direction" column contains the standard meteorological FROM
        # direction, confirming this convention.
        # Negate both to convert to the standard "wind blowing toward" convention
        # used by ERA5 (u_east > 0 = blowing east, v_north > 0 = blowing north).
        northName = f"{int(alt)}m X-wind (m/s)"
        eastName  = f"{int(alt)}m Y-wind (m/s)"
        if eastName in df.columns and northName in df.columns:
            eastCols.append(eastName)
            northCols.append(northName)
            validAlt.append(alt)

    if not eastCols:
        raise ValueError(
            f"No X-wind / Y-wind columns found for altitudes {altitudes}. "
            f"Available columns: {list(df.columns[:20])}"
        )

    windEast  = -df[eastCols].values.astype(float)
    windNorth = -df[northCols].values.astype(float)
    return windEast, windNorth, np.array(validAlt)


def read_data(config=None):
    """Read lidar wind data from RTD files and combine into a single dataset.

    Reads all WindCube WLS7-130 .rtd files in the specified directory and
    combines them into a single dataset compatible with the wind profile
    clustering pipeline.

    Args:
        config (dict, optional): Configuration dictionary with optional keys:
            - 'data_dir' (str or Path): Directory containing .rtd files.
                Defaults to 'data/WSL7-130_lidar' relative to the project root.
            - 'date_range' (tuple): (startDate, endDate) as 'YYYY-MM-DD' strings.
            - 'altitudes' (list): Altitude gates to extract [m]. Uses all by default.
            - 'resample_hourly' (bool): If True, resample the ~4-second data to
                hourly means (dropping hours with any NaN). Default False.

    Returns:
        dict: Dataset containing:
            - 'wind_speed_east' (ndarray): East component [m/s] (nSamples x nAlt).
            - 'wind_speed_north' (ndarray): North component [m/s] (nSamples x nAlt).
            - 'n_samples' (int): Number of time samples.
            - 'datetime' (ndarray): Datetime values.
            - 'altitude' (ndarray): Altitude gates [m].
            - 'years' (tuple): (firstYear, lastYear).
    """
    if config is None:
        config = {}

    defaultDataDir = Path(__file__).parent.parent.parent.parent / 'data' / 'WSL7-130_lidar'
    dataDir = Path(config.get('data_dir', defaultDataDir))
    dateRange = config.get('date_range', None)
    targetAltitudes = config.get('altitudes', None)
    resampleHourly = config.get('resample_hourly', False)

    rtdFiles = sorted(glob(str(dataDir / '*.rtd')))
    if not rtdFiles:
        raise FileNotFoundError(f"No .rtd files found in {dataDir}")

    print(f"Found {len(rtdFiles)} RTD files in {dataDir}")

    # Optional date filtering based on the date embedded in the filename
    if dateRange is not None:
        startDate = pd.Timestamp(dateRange[0])
        endDate = pd.Timestamp(dateRange[1])
        filtered = []
        for fp in rtdFiles:
            match = re.search(r'(\d{4})_(\d{2})_(\d{2})', Path(fp).stem)
            if match:
                fileDate = pd.Timestamp(
                    f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                )
                if startDate <= fileDate <= endDate:
                    filtered.append(fp)
            else:
                filtered.append(fp)  # include if we cannot parse the date
        rtdFiles = filtered
        print(f"Filtered to {len(rtdFiles)} files ({dateRange[0]} to {dateRange[1]})")

    if not rtdFiles:
        raise ValueError("No files remain after date filtering")

    allEast, allNorth, allDatetime = [], [], []
    commonAltitudes = None

    for i, fp in enumerate(rtdFiles):
        print(f"  [{i + 1}/{len(rtdFiles)}] {Path(fp).name}")
        df, metadata = read_rtd_file(fp)
        if df is None or df.empty:
            continue

        fileAltitudes = metadata.get('altitudes')
        if fileAltitudes is None:
            continue

        # Establish the shared altitude grid from the first successful file
        if commonAltitudes is None:
            commonAltitudes = (
                np.array(targetAltitudes, dtype=float)
                if targetAltitudes is not None
                else fileAltitudes
            )

        try:
            windEast, windNorth, validAlt = extract_wind_components(df, commonAltitudes)
        except Exception as e:
            warnings.warn(f"Skipping {Path(fp).name}: {e}")
            continue

        # If some altitudes are missing, pad with NaN
        if not np.array_equal(validAlt, commonAltitudes):
            nT = len(df)
            fullEast = np.full((nT, len(commonAltitudes)), np.nan)
            fullNorth = np.full((nT, len(commonAltitudes)), np.nan)
            for j, alt in enumerate(commonAltitudes):
                idx = np.where(validAlt == alt)[0]
                if len(idx) > 0:
                    fullEast[:, j] = windEast[:, idx[0]]
                    fullNorth[:, j] = windNorth[:, idx[0]]
            windEast, windNorth = fullEast, fullNorth

        allEast.append(windEast)
        allNorth.append(windNorth)
        allDatetime.extend(df.index.tolist())

    if not allEast:
        raise ValueError("No data could be read from any RTD files")

    combinedEast = np.concatenate(allEast, axis=0)
    combinedNorth = np.concatenate(allNorth, axis=0)
    combinedDatetime = np.array(allDatetime)

    # Sort chronologically
    sortIdx = np.argsort(combinedDatetime)
    combinedEast = combinedEast[sortIdx]
    combinedNorth = combinedNorth[sortIdx]
    combinedDatetime = combinedDatetime[sortIdx]
    # Optional hourly resampling (reduces ~4-second data to hourly means)
    if resampleHourly:
        print("Resampling to hourly means...")
        dtIndex = pd.DatetimeIndex(combinedDatetime)
        eastDf  = pd.DataFrame(combinedEast,  index=dtIndex)
        northDf = pd.DataFrame(combinedNorth, index=dtIndex)
        eastHourly  = eastDf.resample('1h').mean()
        northHourly = northDf.resample('1h').mean()
        # Drop hours where any altitude gate has a NaN
        valid = eastHourly.notna().all(axis=1) & northHourly.notna().all(axis=1)
        eastHourly  = eastHourly.loc[valid]
        northHourly = northHourly.loc[valid]
        combinedEast     = eastHourly.values
        combinedNorth    = northHourly.values
        combinedDatetime = np.array(eastHourly.index)
    yearsIdx = pd.DatetimeIndex(combinedDatetime).year
    yearRange = (int(yearsIdx.min()), int(yearsIdx.max()))

    nSamples = len(combinedDatetime)
    print(f"\nTotal samples : {nSamples}")
    print(f"Altitude [m]  : {commonAltitudes}")
    print(f"Time range    : {combinedDatetime[0]} -> {combinedDatetime[-1]}")
    print(f"Years         : {yearRange[0]}-{yearRange[1]}")

    return {
        'wind_speed_east': combinedEast,
        'wind_speed_north': combinedNorth,
        'n_samples': nSamples,
        'datetime': combinedDatetime,
        'altitude': commonAltitudes,
        'years': yearRange,
    }
