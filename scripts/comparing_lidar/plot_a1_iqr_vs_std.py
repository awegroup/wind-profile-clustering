"""Standalone report figure: Analysis 1 — IQR vs std ratio (lidar / ERA5).

Replicates the A1 diagnostic from std_diagnostic.py as a single, clean figure
suitable for inclusion in a report.

Saves:  results/plots/a1_iqr_vs_std.pdf
"""

import sys
import warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

srcPath = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(srcPath))

from wind_profile_clustering.read_data.era5 import read_data as read_era5
from wind_profile_clustering.read_data.wls7_130_lidar import read_rtd_file

# ── Configuration ──────────────────────────────────────────────────────────────
LIDAR_LAT      = 54.1254
LIDAR_LON      = -9.7801
LIDAR_DATA_DIR = Path('data/WSL7-130_lidar')
ERA5_DATA_DIR  = 'data/era5'
ERA5_YEARS     = (2024, 2025)
LIDAR_ALTS     = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 250]
DATE_RANGE     = ('2025-01-08', '2025-06-11')


# ── Helpers ────────────────────────────────────────────────────────────────────

def _era5_longitude(era5_data_dir, target_lon):
    files = sorted(Path(era5_data_dir).glob('ml_*.netcdf'))
    if not files:
        return target_lon
    try:
        ds = xr.open_dataset(files[0])
        lon_max = float(ds['longitude'].values.max())
        ds.close()
        if lon_max > 180.0:
            return float(target_lon % 360)
    except Exception:
        pass
    return target_lon


def read_raw_lidar(dataDir, altitudes):
    """Read all RTD files and return raw u/v arrays.

    Args:
        dataDir (str or Path): Directory containing .rtd files.
        altitudes (list): Altitude gates to extract [m].

    Returns:
        dict: Keys 'u', 'v' (nSamples x nAlt), 'datetime'.
    """
    dataDir   = Path(dataDir)
    altitudes = np.array(altitudes, dtype=float)
    rtdFiles  = sorted(glob(str(dataDir / '*.rtd')))

    allU, allV, allDt = [], [], []
    for fp in rtdFiles:
        df, _meta = read_rtd_file(fp)
        if df is None or df.empty:
            continue
        nT = len(df)
        u  = np.full((nT, len(altitudes)), np.nan)
        v  = np.full((nT, len(altitudes)), np.nan)
        for j, alt in enumerate(altitudes):
            eastCol  = f"{int(alt)}m Y-wind (m/s)"
            northCol = f"{int(alt)}m X-wind (m/s)"
            if eastCol  in df.columns: u[:, j] = -df[eastCol].values.astype(float)
            if northCol in df.columns: v[:, j] = -df[northCol].values.astype(float)
        allU.append(u);  allV.append(v)
        allDt.extend(df.index.tolist())

    u   = np.concatenate(allU, axis=0)
    v   = np.concatenate(allV, axis=0)
    dts = np.array(allDt)
    idx = np.argsort(dts)
    return {'u': u[idx], 'v': v[idx], 'datetime': dts[idx]}


def resample_lidar(u, v, datetimes, altitudes, freq='1h'):
    """Resample u/v to hourly means.

    Args:
        u (ndarray): East component (nSamples x nAlt).
        v (ndarray): North component (nSamples x nAlt).
        datetimes (array-like): Sample datetimes.
        altitudes (array-like): Column labels.
        freq (str): Pandas frequency string.

    Returns:
        tuple: (uH, vH) DataFrames.
    """
    dtIdx = pd.DatetimeIndex(datetimes)
    uH = pd.DataFrame(u.astype(float), index=dtIdx, columns=altitudes).resample(freq).mean()
    vH = pd.DataFrame(v.astype(float), index=dtIdx, columns=altitudes).resample(freq).mean()
    return uH, vH


def match_datasets(e5_u, e5_v, lid_u, lid_v, dateRange):
    """Align ERA5 and lidar on common valid timestamps within dateRange.

    Args:
        e5_u (DataFrame): ERA5 east component.
        e5_v (DataFrame): ERA5 north component.
        lid_u (DataFrame): Lidar east component.
        lid_v (DataFrame): Lidar north component.
        dateRange (tuple): ('YYYY-MM-DD', 'YYYY-MM-DD').

    Returns:
        tuple: (e5U, e5V, lidU, lidV) float arrays (nH x nAlt), nHours.
    """
    cIdx = e5_u.index.intersection(lid_u.index)
    cIdx = cIdx[
        (cIdx >= pd.Timestamp(dateRange[0])) &
        (cIdx <  pd.Timestamp(dateRange[1]) + pd.Timedelta('1D'))
    ]
    e5U  = e5_u.loc[cIdx];  e5V  = e5_v.loc[cIdx]
    lidU = lid_u.loc[cIdx]; lidV = lid_v.loc[cIdx]
    ok   = e5U.notna().all(axis=1) & lidU.notna().all(axis=1)
    n    = int(ok.sum())
    return (e5U.loc[ok].values.astype(float),
            e5V.loc[ok].values.astype(float),
            lidU.loc[ok].values.astype(float),
            lidV.loc[ok].values.astype(float), n)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    """Compute A1 and save standalone report figure."""
    warnings.filterwarnings('ignore')
    outDir = Path('results/plots')
    outDir.mkdir(parents=True, exist_ok=True)

    alts = np.array(LIDAR_ALTS)

    # ── Read ERA5 ──────────────────────────────────────────────────────────────
    era5_lon = _era5_longitude(ERA5_DATA_DIR, LIDAR_LON)
    print("Reading ERA5 ...")
    era5_raw = read_era5({
        'data_dir':       ERA5_DATA_DIR,
        'location':       {'latitude': LIDAR_LAT, 'longitude': era5_lon},
        'altitude_range': (40, 250),
        'years':          ERA5_YEARS,
    })
    era5_alts = [int(round(a)) for a in era5_raw['altitude']]
    era5_dt   = pd.DatetimeIndex(era5_raw['datetime']).floor('h')
    era5_u1   = pd.DataFrame(era5_raw['wind_speed_east'],  index=era5_dt,
                              columns=era5_alts)[LIDAR_ALTS]
    era5_v1   = pd.DataFrame(era5_raw['wind_speed_north'], index=era5_dt,
                              columns=era5_alts)[LIDAR_ALTS]

    # ── Read lidar ─────────────────────────────────────────────────────────────
    print("Reading lidar ...")
    lidar_raw        = read_raw_lidar(LIDAR_DATA_DIR, LIDAR_ALTS)
    lid_u1, lid_v1   = resample_lidar(lidar_raw['u'], lidar_raw['v'],
                                      lidar_raw['datetime'], alts, '1h')

    # ── Match and compute statistics ───────────────────────────────────────────
    e5U, e5V, lidU, lidV, nHours = match_datasets(
        era5_u1, era5_v1, lid_u1, lid_v1, DATE_RANGE)

    spd_e5  = np.sqrt(e5U**2  + e5V**2)
    spd_lid = np.sqrt(lidU**2 + lidV**2)

    e5_std  = np.array([np.nanstd(spd_e5[:, i])  for i in range(len(alts))])
    lid_std = np.array([np.nanstd(spd_lid[:, i]) for i in range(len(alts))])
    e5_iqr  = np.array([np.nanpercentile(spd_e5[:, i],  75)
                         - np.nanpercentile(spd_e5[:, i],  25) for i in range(len(alts))])
    lid_iqr = np.array([np.nanpercentile(spd_lid[:, i], 75)
                         - np.nanpercentile(spd_lid[:, i], 25) for i in range(len(alts))])

    print(f"Matched hours: {nHours}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4, 5.5))

    BLUE   = '#0072B2'
    ORANGE = '#D55E00'

    ax.plot(lid_std / e5_std, alts, 'o-',
            color=BLUE,   lw=1.5, ms=5, label='Std ratio  (lidar / ERA5)')
    ax.plot(lid_iqr / e5_iqr, alts, 's--',
            color=ORANGE, lw=1.5, ms=5, label='IQR ratio  (lidar / ERA5)')
    ax.axvline(1.0, color='#666666', ls=':', lw=1.2, label='Ratio = 1')

    ax.set_xlabel('Lidar / ERA5 ratio', fontsize=11)
    ax.set_ylabel('Altitude (m AGL)', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.set_ylim(20, 265)
    ax.set_xlim(left=0.9)
    ax.set_yticks(LIDAR_ALTS)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savePath = outDir / 'a1_iqr_vs_std.pdf'
    fig.savefig(savePath, bbox_inches='tight')
    print(f"Saved: {savePath}")
    plt.close(fig)


if __name__ == '__main__':
    main()
