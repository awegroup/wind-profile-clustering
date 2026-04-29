"""Combined ERA5 and WLS7-130 lidar wind analysis at Bangor Erris.

Reads ERA5 and lidar data once and produces the following plots:
  1. Lidar data availability
  2. Mean wind speed comparison  (ERA5 vs lidar, all 12 lidar altitude gates)
  3. Weibull distribution comparison  (100 m and 200 m)
  4. Wind rose comparison  (100 m and 200 m)
  5. Mean wind speed profile comparison
  6. Logarithmic profile fitting comparison
  7. Wind profile clustering comparison  (for each n in N_CLUSTERS_LIST)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from scipy import stats

srcPath = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(srcPath))

from wind_profile_clustering.read_data.era5 import read_data as read_era5
from wind_profile_clustering.read_data.wls7_130_lidar import read_data as read_lidar
from wind_profile_clustering.clustering import perform_clustering_analysis, predict_cluster
from wind_profile_clustering.preprocess_data import preprocess_data
from wind_profile_clustering.fitting_and_prescribing.fit_profile import fit_wind_profile

# =============================================================================
# USER CONFIGURATION
# =============================================================================
LIDAR_LAT         = 54.1254
LIDAR_LON         = -9.7801
LIDAR_DATA_DIR    = 'data/WSL7-130_lidar'
ERA5_DATA_DIR     = 'data/era5'
ERA5_YEARS        = (2024, 2025)
LIDAR_ALTS        = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 250]
WEIBULL_ROSE_ALTS = [100, 200]
DATE_RANGE        = ('2025-01-08', '2025-06-11')   # comparison period
N_CLUSTERS        = 8
REF_HEIGHT        = 100.0
SAVE_PLOTS        = True
# =============================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _era5_longitude(era5_data_dir, target_lon):
    """Detect ERA5 longitude convention (0-360 vs -180-180).

    Args:
        era5_data_dir (str or Path): Directory containing ERA5 NetCDF files.
        target_lon (float): Desired longitude in -180...180 convention.

    Returns:
        float: Longitude adjusted to match the ERA5 file convention.
    """
    files = sorted(Path(era5_data_dir).glob('ml_*.netcdf'))
    if not files:
        return target_lon
    try:
        ds = xr.open_dataset(files[0])
        lon_coord = 'longitude' if 'longitude' in ds else 'lon'
        lon_max = float(ds[lon_coord].values.max())
        ds.close()
        if lon_max > 180.0:
            return float(target_lon % 360)
    except Exception:
        pass
    return target_lon


def fit_weibull(speeds):
    """Fit a 2-parameter Weibull (loc = 0) to positive wind speeds.

    Args:
        speeds (ndarray): Wind speed samples.

    Returns:
        tuple: (k, c) shape and scale parameters.
    """
    v = speeds[~np.isnan(speeds) & (speeds > 0)]
    k, _loc, c = stats.weibull_min.fit(v, floc=0)
    return float(k), float(c)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_data_availability(data, savePath=None):
    """Plot daily data availability as a bar chart.

    Args:
        data (dict): Output of read_lidar().
        savePath (str or Path, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    datetimes = pd.DatetimeIndex(data['datetime'])
    midIdx = len(data['altitude']) // 2
    validMask = ~np.isnan(data['wind_speed_east'][:, midIdx])

    dfValid = pd.Series(validMask.astype(int), index=datetimes)
    dailyValid = dfValid.resample('D').sum()
    dailyTotal = pd.Series(1, index=datetimes).resample('D').count()
    availability = (dailyValid / dailyTotal * 100).clip(upper=100)

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.bar(availability.index, availability.values, width=0.8, color='#0072B2', alpha=0.8)
    ax.vlines([pd.Timestamp(DATE_RANGE[0]), pd.Timestamp(DATE_RANGE[1])], ymin=0, ymax=108, color='#E69F00', ls='--', lw=2, label='Comparison period')
    ax.set_xlabel('Date')
    ax.set_ylabel('Valid scans (%)')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 108)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()

    if savePath:
        fig.savefig(savePath)
        print(f"Saved: {savePath}")
        plt.close(fig)
    return fig


def plot_mean_wind_speed(era5_by_alt, lidar_by_alt, altitudes, n_hours,
                         savePath=None):
    """Bar chart comparing mean (+-1 std) wind speed at each altitude.

    Error bars (+-1 std) are described in the legend labels.

    Args:
        era5_by_alt (dict): ERA5 wind speed arrays keyed by altitude [m].
        lidar_by_alt (dict): Lidar wind speed arrays keyed by altitude [m].
        altitudes (list): Altitude levels to include [m].
        n_hours (int): Number of matched hours.
        savePath (str or Path, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    x     = np.arange(len(altitudes))
    width = 0.35

    era5_means  = [np.nanmean(era5_by_alt[a])  for a in altitudes]
    era5_stds   = [np.nanstd(era5_by_alt[a])   for a in altitudes]
    lidar_means = [np.nanmean(lidar_by_alt[a]) for a in altitudes]
    lidar_stds  = [np.nanstd(lidar_by_alt[a])  for a in altitudes]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, era5_means,  width, yerr=era5_stds,
           label='ERA5 (mean \u00b11 std)', color='#0072B2', alpha=0.85, capsize=4)
    ax.bar(x + width / 2, lidar_means, width, yerr=lidar_stds,
           label='WLS7-130 lidar (mean \u00b11 std)', color='#D55E00', alpha=0.85, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(a)} m' for a in altitudes], rotation=45, ha='right')
    ax.set_xlabel('Altitude')
    ax.set_ylabel('Wind speed (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    if savePath:
        fig.savefig(savePath)
        print(f"Saved: {savePath}")
        plt.close(fig)
    return fig


def plot_weibull_comparison(era5_by_alt, lidar_by_alt, altitudes, n_hours,
                             savePath=None):
    """Histogram + fitted Weibull PDF comparison at selected altitudes.

    Args:
        era5_by_alt (dict): ERA5 wind speed arrays keyed by altitude [m].
        lidar_by_alt (dict): Lidar wind speed arrays keyed by altitude [m].
        altitudes (list): Altitude levels to include [m].
        n_hours (int): Number of matched hours.
        savePath (str or Path, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    n_alts = len(altitudes)
    fig, axes = plt.subplots(1, n_alts, figsize=(4 * n_alts, 4))
    if n_alts == 1:
        axes = [axes]

    for ax, alt in zip(axes, altitudes):
        v_era5  = era5_by_alt[alt]
        v_lidar = lidar_by_alt[alt]

        v_era5_c  = v_era5[~np.isnan(v_era5)   & (v_era5  > 0)]
        v_lidar_c = v_lidar[~np.isnan(v_lidar)  & (v_lidar > 0)]

        k_e, c_e = fit_weibull(v_era5_c)
        k_l, c_l = fit_weibull(v_lidar_c)

        vmax     = max(v_era5_c.max(), v_lidar_c.max())
        binEdges = np.linspace(0, vmax, 40)
        vGrid    = np.linspace(0.01, vmax, 300)

        ax.hist(v_era5_c,  bins=binEdges, density=True, alpha=0.35,
                color='#0072B2', label='ERA5')
        ax.hist(v_lidar_c, bins=binEdges, density=True, alpha=0.35,
                color='#D55E00', label='WLS7-130 lidar')
        ax.plot(vGrid, stats.weibull_min.pdf(vGrid, k_e, loc=0, scale=c_e),
                '-', color='#0072B2', lw=2,
                label=f'ERA5 Weibull\nk={k_e:.2f},  c={c_e:.2f} m/s')
        ax.plot(vGrid, stats.weibull_min.pdf(vGrid, k_l, loc=0, scale=c_l),
                '-', color='#D55E00', lw=2,
                label=f'Lidar Weibull\nk={k_l:.2f},  c={c_l:.2f} m/s')

        ax.set_xlabel('Wind speed (m/s)')
        ax.set_ylabel('Probability density')
        ax.set_title(f'{int(alt)} m')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if savePath:
        fig.savefig(savePath, bbox_inches='tight')
        print(f"Saved: {savePath}")
        plt.close(fig)
    return fig


def plot_wind_rose_comparison(era5_uv_by_alt, lidar_uv_by_alt, altitudes, n_hours,
                              savePath=None):
    """Side-by-side wind rose comparison at selected altitudes.

    Args:
        era5_uv_by_alt (dict): ERA5 (u, v) tuples keyed by altitude [m].
        lidar_uv_by_alt (dict): Lidar (u, v) tuples keyed by altitude [m].
        altitudes (list): Altitude levels to include [m].
        n_hours (int): Number of matched hours.
        savePath (str or Path, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    N_SECTORS    = 16
    SECTOR_SIZE  = 360.0 / N_SECTORS
    SPEED_BINS   = [0, 4, 8, 12, 16, np.inf]
    SPEED_LABELS = ['0-4', '4-8', '8-12', '12-16', '>16']
    SPEED_COLORS = ['#4575b4', '#74add1', '#fee090', '#f46d43', '#d73027']

    sector_centers = np.radians(np.arange(N_SECTORS) * SECTOR_SIZE)
    sector_width   = np.radians(SECTOR_SIZE) * 0.9

    def _rose_freqs(u, v):
        direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
        speed     = np.sqrt(u**2 + v**2)
        valid     = ~(np.isnan(direction) | np.isnan(speed))
        direction, speed = direction[valid], speed[valid]
        if len(direction) == 0:
            return np.zeros((N_SECTORS, len(SPEED_BINS) - 1))
        sector_idx = ((direction + SECTOR_SIZE / 2) % 360 / SECTOR_SIZE).astype(int) % N_SECTORS
        freqs = np.zeros((N_SECTORS, len(SPEED_BINS) - 1))
        for i in range(N_SECTORS):
            mask = sector_idx == i
            if not mask.any():
                continue
            s = speed[mask]
            for j in range(len(SPEED_BINS) - 1):
                freqs[i, j] = np.sum((s >= SPEED_BINS[j]) & (s < SPEED_BINS[j + 1]))
        freqs /= len(direction)
        return freqs

    n_alts = len(altitudes)
    fig, axes = plt.subplots(
        n_alts, 2,
        figsize=(8, 2.5 * n_alts),
        subplot_kw={'projection': 'polar'},
    )
    if n_alts == 1:
        axes = axes[np.newaxis, :]

    # Pre-compute all sector frequencies to determine a shared radial scale
    allFreqs = [
        (_rose_freqs(era5_uv_by_alt[alt][0],  era5_uv_by_alt[alt][1]),
         _rose_freqs(lidar_uv_by_alt[alt][0], lidar_uv_by_alt[alt][1]))
        for alt in altitudes
    ]
    globalMax = max(
        float(f.sum(axis=1).max())
        for pair in allFreqs
        for f in pair
    )
    radialMax = globalMax * 1.1

    for row, alt in enumerate(altitudes):
        for col in range(2):
            ax    = axes[row, col]
            freqs = allFreqs[row][col]
            title = (f'ERA5 - {int(alt)} m'
                     if col == 0 else f'WLS7-130 lidar - {int(alt)} m')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)

            bottom = np.zeros(N_SECTORS)
            for j, (color, label) in enumerate(zip(SPEED_COLORS, SPEED_LABELS)):
                ax.bar(
                    sector_centers, freqs[:, j],
                    width=sector_width,
                    bottom=bottom,
                    color=color, alpha=0.85,
                    label=label if row == 0 and col == 0 else None,
                )
                bottom += freqs[:, j]

            ax.set_ylim(0, radialMax)
            ax.set_title(title, pad=12)
            ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%')
            )

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85)
               for c in SPEED_COLORS]
    leg = fig.legend(handles, [f'{l} m/s' for l in SPEED_LABELS],
                     title='Wind speed', loc='lower center',
                     ncol=len(SPEED_LABELS), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    # Lower legend by 1.5 × its own height so it does not overlap the wind rose axes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg_h_frac = (leg.get_window_extent(renderer).height
                  / fig.get_window_extent(renderer).height)
    leg.set_bbox_to_anchor((0.5, -1.5 * leg_h_frac), transform=fig.transFigure)

    if savePath:
        fig.savefig(savePath, bbox_inches='tight')
        print(f"Saved: {savePath}")
        plt.close(fig)
    return fig


def plot_mean_profiles(era5_data, lidar_data, n_hours, savePath=None):
    """Compare mean wind speed profiles for ERA5 and lidar.

    Args:
        era5_data (dict): Wind data dict for ERA5 matched samples.
        lidar_data (dict): Wind data dict for lidar matched samples.
        n_hours (int): Number of matched hours.
        savePath (str or Path, optional): Path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    alts = era5_data['altitude']

    era5_spd  = np.sqrt(era5_data['wind_speed_east']**2
                        + era5_data['wind_speed_north']**2)
    lidar_spd = np.sqrt(lidar_data['wind_speed_east']**2
                        + lidar_data['wind_speed_north']**2)

    era5_mean,  era5_std  = np.nanmean(era5_spd,  axis=0), np.nanstd(era5_spd,  axis=0)
    lidar_mean, lidar_std = np.nanmean(lidar_spd, axis=0), np.nanstd(lidar_spd, axis=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(era5_mean,  alts, '-o', color='#0072B2', lw=1.5, ms=4, label='ERA5')
    ax.fill_betweenx(alts, era5_mean - era5_std, era5_mean + era5_std,
                     color='#0072B2', alpha=0.15)
    ax.plot(lidar_mean, alts, '-o', color='#D55E00', lw=1.5, ms=4, label='WLS7-130 lidar')
    ax.fill_betweenx(alts, lidar_mean - lidar_std, lidar_mean + lidar_std,
                     color='#D55E00', alpha=0.15)

    ax.set_xlabel('Mean wind speed (m/s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_ylim(0, 250)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if savePath:
        fig.savefig(savePath)
        print(f"Saved: {savePath}")
        plt.close(fig)
    return fig


def plot_log_fit_comparison(era5_data, lidar_data, n_hours, savePath=None):
    """Compare logarithmic profile fits for ERA5 and lidar.

    Uses fit_wind_profile from the fitting_and_prescribing module.

    Args:
        era5_data (dict): Wind data dict for ERA5 matched samples.
        lidar_data (dict): Wind data dict for lidar matched samples.
        n_hours (int): Number of matched hours.
        savePath (str or Path, optional): Path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    alts = era5_data['altitude']

    era5_spd  = np.sqrt(era5_data['wind_speed_east']**2
                        + era5_data['wind_speed_north']**2)
    lidar_spd = np.sqrt(lidar_data['wind_speed_east']**2
                        + lidar_data['wind_speed_north']**2)
    era5_mean  = np.nanmean(era5_spd,  axis=0)
    lidar_mean = np.nanmean(lidar_spd, axis=0)

    era5_fit  = fit_wind_profile(era5_data,  profileType='logarithmic',
                                  refHeight=REF_HEIGHT)
    lidar_fit = fit_wind_profile(lidar_data, profileType='logarithmic',
                                  refHeight=REF_HEIGHT)

    era5_ref_spd  = float(np.nanmean(era5_fit['normalisationWindSpeeds']))
    lidar_ref_spd = float(np.nanmean(lidar_fit['normalisationWindSpeeds']))
    era5_fitted   = era5_fit['prl'][0]  * era5_ref_spd
    lidar_fitted  = lidar_fit['prl'][0] * lidar_ref_spd

    ep = era5_fit['fitParams']
    lp = lidar_fit['fitParams']

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.plot(era5_mean,    alts, 'o',  color='#0072B2', ms=5, label='ERA5 mean')
    ax.plot(era5_fitted,  alts, '-',  color='#0072B2', lw=1.5,
            label=f'ERA5 log fit  u*={ep["friction_velocity"]:.3f} m/s,'
                  f'  z0={ep["roughness_length"]:.4f} m')
    ax.plot(lidar_mean,   alts, 's',  color='#D55E00', ms=5, label='Lidar mean')
    ax.plot(lidar_fitted, alts, '-',  color='#D55E00', lw=1.5,
            label=f'Lidar log fit  u*={lp["friction_velocity"]:.3f} m/s,'
                  f'  z0={lp["roughness_length"]:.4f} m')

    ax.set_xlabel('Wind speed (m/s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_ylim(0, 250)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if savePath:
        fig.savefig(savePath)
        print(f"Saved: {savePath}")
        plt.close(fig)
    return fig


def _compute_cluster_xlim(clust_result, alts, refHeight):
    """Return (xmin, xmax) of the scaled parallel profiles for a cluster result.

    Args:
        clust_result (dict): Return value of perform_clustering_analysis.
        alts (ndarray): Altitude array [m].
        refHeight (float): Reference height used for normalisation.

    Returns:
        tuple: (xmin, xmax) floats.
    """
    prl = clust_result['clusteringResults']['clusters_feature']['parallel']
    prp = clust_result['clusteringResults']['clusters_feature']['perpendicular']
    prl_scaled = []
    for u, v in zip(prl, prp):
        uRef = float(np.interp(refHeight, alts, u))
        vRef = float(np.interp(refHeight, alts, v))
        wRef = np.sqrt(uRef**2 + vRef**2)
        sf   = 1.0 / wRef if wRef != 0 else 1.0
        prl_scaled.append(u * sf)
    all_vals = np.concatenate(prl_scaled)
    return float(all_vals.min()), float(all_vals.max())


def plot_era5_clusters(era5_result, alts, refHeight=200.0, datasetLabel='ERA5',
                       xlim=None, savePath=None):
    """Plot cluster profiles: parallel component.

    Clusters are sorted by ascending power-law shear exponent.  Each cluster
    label includes the fitted exponent.

    Args:
        era5_result (dict): Return value of perform_clustering_analysis.
        alts (ndarray): Altitude array [m].
        refHeight (float): Reference height for normalisation. Defaults to 200.0.
        datasetLabel (str): Dataset name shown in the subplot titles.
        xlim (tuple, optional): (xmin, xmax) for the parallel-component axis.
        savePath (str or Path, optional): Path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    prl  = era5_result['clusteringResults']['clusters_feature']['parallel']
    prp  = era5_result['clusteringResults']['clusters_feature']['perpendicular']
    freq = np.array(era5_result['clusteringResults']['frequency_clusters'])
    n_clusters = len(freq)

    colors      = plt.rcParams['axes.prop_cycle'].by_key()['color']
    line_styles = ['-', '--', ':', '-.']
    markers     = ['o', 's', '^', 'D']

    # Scale each cluster so prl = 1 at reference height
    prl_scaled = []
    prp_scaled = []
    for u, v in zip(prl, prp):
        uRef = float(np.interp(refHeight, alts, u))
        vRef = float(np.interp(refHeight, alts, v))
        wRef = np.sqrt(uRef**2 + vRef**2)
        sf   = 1.0 / wRef if wRef != 0 else 1.0
        prl_scaled.append(u * sf)
        prp_scaled.append(v * sf)

    # Fit power-law shear exponent per cluster via OLS in log-log space
    lnZ   = np.log(np.maximum(alts / refHeight, 1e-9))
    denom = float(np.sum(lnZ ** 2))
    alphas = np.array([
        float(np.sum(lnZ * np.log(np.maximum(ps, 1e-9))) / denom)
        if denom > 0 else 0.0
        for ps in prl_scaled
    ])

    # Sort clusters by descending frequency (most common first)
    sortIdx    = np.argsort(-freq)
    prl_scaled = [prl_scaled[i] for i in sortIdx]
    prp_scaled = [prp_scaled[i] for i in sortIdx]
    freq       = freq[sortIdx]
    alphas     = alphas[sortIdx]

    print(f'  {datasetLabel} clusters (sorted by descending frequency):')
    for k, (a, f) in enumerate(zip(alphas, freq)):
        print(f'    Cluster {k + 1}: alpha = {a:.3f}  ({f:.1f} %)')

    fig, ax_prl = plt.subplots(figsize=(8, 4))

    for k in range(n_clusters):
        color     = colors[k % len(colors)]
        style_idx = k // len(colors)
        ls        = line_styles[style_idx % len(line_styles)]
        marker    = markers[style_idx % len(markers)]
        fmt       = ls + marker
        label     = f'Cluster {k + 1}  ({freq[k]:.1f} %)  \u03b1 = {alphas[k]:.2f}'
        ax_prl.plot(prl_scaled[k], alts, fmt, color=color, lw=1.5, ms=3, label=label)

    ax_prl.set_xlabel('Parallel component (-)')
    ax_prl.set_ylabel('Altitude (m)')
    ax_prl.set_ylim(0, 250)
    if xlim is not None:
        ax_prl.set_xlim(xlim)
    ax_prl.set_title(f'{datasetLabel} — Parallel')
    ax_prl.legend(fontsize=7)
    ax_prl.grid(True, alpha=0.3)

    fig.tight_layout()

    if savePath:
        fig.savefig(savePath)
        print(f'Saved: {savePath}')
        plt.close(fig)
    return fig, sortIdx


def plot_cluster_frequency_comparison(era5_freq, lidar_freq, savePath=None):
    """Grouped bar chart comparing ERA5 and lidar cluster occurrence frequencies.

    Args:
        era5_freq (ndarray): ERA5 cluster frequencies [%], length n_clusters.
        lidar_freq (ndarray): Lidar cluster frequencies [%], length n_clusters.
        savePath (str or Path, optional): Path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    n_clusters = len(era5_freq)
    x     = np.arange(1, n_clusters + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.bar(x - width / 2, era5_freq,  width,
           label='ERA5',            color='#0072B2', alpha=0.85)
    ax.bar(x + width / 2, lidar_freq, width,
           label='WLS7-130 lidar',  color='#D55E00', alpha=0.85)

    for xi, (ef, lf) in zip(x, zip(era5_freq, lidar_freq)):
        ax.text(xi - width / 2, ef, f'{ef:.1f}', ha='center', va='bottom', fontsize=7)
        ax.text(xi + width / 2, lf, f'{lf:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Frequency (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()

    if savePath:
        fig.savefig(savePath)
        print(f'Saved: {savePath}')
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Profile shape analysis helpers and plot
# ---------------------------------------------------------------------------

def _compute_profile_metrics(uArr, vArr, alts, refHeight=200.0):
    """Compute power-law shear exponent per wind profile.

    Args:
        uArr (ndarray): East component (nSamples x nAlt).
        vArr (ndarray): North component (nSamples x nAlt).
        alts (ndarray): Altitude gates [m].
        refHeight (float): Reference height for power-law normalisation [m].

    Returns:
        ndarray: alpha array of length nSamples.
    """
    spd    = np.sqrt(uArr**2 + vArr**2)              # (n, nAlt)
    refIdx = int(np.argmin(np.abs(alts - refHeight)))
    lnZ    = np.log(alts / alts[refIdx])             # (nAlt,), = 0 at refHeight

    # Power-law shear exponent: OLS through origin in log-log space
    with np.errstate(divide='ignore', invalid='ignore'):
        lnU = np.log(np.maximum(spd / spd[:, refIdx:refIdx + 1], 1e-9))
    denom = float(np.sum(lnZ ** 2))
    alpha = np.sum(lnZ[np.newaxis, :] * lnU, axis=1) / denom

    # Mask invalid profiles
    badMask        = (spd[:, refIdx] <= 0) | np.any(np.isnan(spd), axis=1)
    alpha[badMask] = np.nan

    return alpha


def plot_profile_shape_analysis(era5_data, lidar_data, n_hours, savePath=None):
    """Compare ERA5 and lidar wind-profile shape distributions independently.

    For each dataset the power-law shear exponent is computed per profile via
    OLS in log-log space.  Each profile is assigned to one of three types:

    * **High shear**   — alpha > 0.30
    * **Normal shear** — 0.10 <= alpha <= 0.30
    * **Low shear**    — alpha < 0.10

    The figure has two panels:

    1. KDE + histogram of alpha for ERA5 vs lidar.
    2. Grouped bar chart of profile-type frequencies.

    Args:
        era5_data (dict): Wind data dict for ERA5 matched samples.
        lidar_data (dict): Wind data dict for lidar matched samples.
        n_hours (int): Number of matched hours.
        savePath (str or Path, optional): Path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    ALPHA_HIGH = 0.30
    ALPHA_LOW  = 0.10

    alts = era5_data['altitude']

    e5Alpha  = _compute_profile_metrics(
        era5_data['wind_speed_east'],  era5_data['wind_speed_north'],
        alts, REF_HEIGHT)
    lidAlpha = _compute_profile_metrics(
        lidar_data['wind_speed_east'], lidar_data['wind_speed_north'],
        alts, REF_HEIGHT)

    # ── Profile-type classification ─────────────────────────────────────────
    def classify(alpha):
        cats = np.full(len(alpha), 'Normal shear', dtype=object)
        cats[alpha > ALPHA_HIGH] = 'High shear'
        cats[alpha < ALPHA_LOW]  = 'Low shear'
        return cats

    e5Cats  = classify(e5Alpha)
    lidCats = classify(lidAlpha)

    catLabels = ['High shear', 'Normal shear', 'Low shear']
    e5Freq  = np.array([np.mean(e5Cats  == c) * 100 for c in catLabels])
    lidFreq = np.array([np.mean(lidCats == c) * 100 for c in catLabels])

    print('  Profile-type frequencies (ERA5 / lidar):')
    for cat, ef, lf in zip(catLabels, e5Freq, lidFreq):
        print(f'    {cat:<14}: ERA5 {ef:5.1f}%  |  lidar {lf:5.1f}%')

    # ── Figure ────────────────────────────────────────────────────────────────
    BLUE   = '#0072B2'
    ORANGE = '#D55E00'
    GREY   = '#888888'

    fig, (ax_alpha, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))

    # -- Panel left: alpha distribution --------------------------------------
    ax = ax_alpha
    validE5  = e5Alpha[~np.isnan(e5Alpha)]
    validLid = lidAlpha[~np.isnan(lidAlpha)]
    binEdges  = np.linspace(-1, 2, 121)
    alphaGrid = np.linspace(-1, 2, 600)
    ax.hist(validE5,  bins=binEdges, density=True, alpha=0.25, color=BLUE,   label='ERA5')
    ax.hist(validLid, bins=binEdges, density=True, alpha=0.25, color=ORANGE, label='WLS7-130 lidar')
    ax.plot(alphaGrid, stats.gaussian_kde(validE5)(alphaGrid),  '-', color=BLUE,   lw=2)
    ax.plot(alphaGrid, stats.gaussian_kde(validLid)(alphaGrid), '-', color=ORANGE, lw=2)
    ax.axvline(ALPHA_LOW,  color=GREY, ls=':',  lw=1.2, label=f'\u03b1 = {ALPHA_LOW}')
    ax.axvline(ALPHA_HIGH, color=GREY, ls='--', lw=1.2, label=f'\u03b1 = {ALPHA_HIGH}')
    ax.set_xlabel('Power-law shear exponent \u03b1 (-)')
    ax.set_ylabel('Probability density')
    ax.set_xlim(-1, 2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- Panel right: profile-type frequency bar chart -----------------------
    ax = ax_bar
    x     = np.arange(len(catLabels))
    width = 0.35
    ax.bar(x - width / 2, e5Freq,  width, label='ERA5',           color=BLUE,   alpha=0.85)
    ax.bar(x + width / 2, lidFreq, width, label='WLS7-130 lidar', color=ORANGE, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(catLabels)
    ax.set_xlabel('Profile type')
    ax.set_ylabel('Frequency (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if savePath:
        fig.savefig(savePath, bbox_inches='tight')
        print(f'Saved: {savePath}')
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Read ERA5 and lidar data once and generate all comparison plots."""
    warnings.filterwarnings('ignore')

    era5_lon = _era5_longitude(ERA5_DATA_DIR, LIDAR_LON)

    # --- Read data (once) ---
    print(f"Reading ERA5 at ({LIDAR_LAT}N, {era5_lon}E) ...")
    era5_raw = read_era5({
        'data_dir': ERA5_DATA_DIR,
        'location': {'latitude': LIDAR_LAT, 'longitude': era5_lon},
        'altitude_range': (40, 250),
        'years': ERA5_YEARS,
    })

    print("\nReading WLS7-130 lidar data...")
    lidar_raw = read_lidar({'data_dir': LIDAR_DATA_DIR})

    # --- Output directory ---
    out_dir = Path(__file__).parent.parent / 'results' / 'plots'
    if SAVE_PLOTS:
        out_dir.mkdir(parents=True, exist_ok=True)

    def _path(name):
        return (out_dir / name) if SAVE_PLOTS else None

    # --- Plot 1: Lidar data availability (uses raw lidar) ---
    print("\n--- Plot 1: Lidar data availability ---")
    plot_data_availability(lidar_raw, savePath=_path('lidar_data_availability.pdf'))

    # --- Align ERA5 and lidar on matched hourly timestamps ---
    print("\nAligning ERA5 and lidar on matched hourly timestamps...")

    era5_alts_all = [int(round(a)) for a in era5_raw['altitude']]
    era5_dt       = pd.DatetimeIndex(era5_raw['datetime']).floor('h')
    lidar_dt      = pd.DatetimeIndex(lidar_raw['datetime'])

    # ERA5 DataFrames restricted to lidar altitude columns
    era5_u_df = pd.DataFrame(era5_raw['wind_speed_east'],  index=era5_dt,
                              columns=era5_alts_all)[LIDAR_ALTS]
    era5_v_df = pd.DataFrame(era5_raw['wind_speed_north'], index=era5_dt,
                              columns=era5_alts_all)[LIDAR_ALTS]

    # Lidar DataFrames -> hourly means
    lidar_u_df = pd.DataFrame(lidar_raw['wind_speed_east'],  index=lidar_dt,
                               columns=LIDAR_ALTS)
    lidar_v_df = pd.DataFrame(lidar_raw['wind_speed_north'], index=lidar_dt,
                               columns=LIDAR_ALTS)
    lidar_u_h = lidar_u_df.resample('1h').mean()
    lidar_v_h = lidar_v_df.resample('1h').mean()

    # Find common timestamps; apply valid mask (all altitudes non-NaN in both)
    common_idx = era5_u_df.index.intersection(lidar_u_h.index)
    # Filter to comparison date range
    common_idx = common_idx[
        (common_idx >= pd.Timestamp(DATE_RANGE[0]))
        & (common_idx < pd.Timestamp(DATE_RANGE[1]) + pd.Timedelta('1D'))
    ]
    era5_u_m  = era5_u_df.loc[common_idx]
    era5_v_m  = era5_v_df.loc[common_idx]
    lidar_u_m = lidar_u_h.loc[common_idx]
    lidar_v_m = lidar_v_h.loc[common_idx]

    valid = era5_u_m.notna().all(axis=1) & lidar_u_m.notna().all(axis=1)
    era5_u_m  = era5_u_m.loc[valid].values.astype(float)
    era5_v_m  = era5_v_m.loc[valid].values.astype(float)
    lidar_u_m = lidar_u_m.loc[valid].values.astype(float)
    lidar_v_m = lidar_v_m.loc[valid].values.astype(float)
    n_hours   = int(valid.sum())

    print(f"Matched hours (all altitudes valid): {n_hours}")

    alts = np.array(LIDAR_ALTS, dtype=float)

    # Speed and component dicts keyed by altitude for bar chart / Weibull / rose
    era5_spd_m  = np.sqrt(era5_u_m**2  + era5_v_m**2)
    lidar_spd_m = np.sqrt(lidar_u_m**2 + lidar_v_m**2)

    era5_by_alt     = {a: era5_spd_m[:, i]               for i, a in enumerate(LIDAR_ALTS)}
    lidar_by_alt    = {a: lidar_spd_m[:, i]              for i, a in enumerate(LIDAR_ALTS)}
    era5_uv_by_alt  = {a: (era5_u_m[:, i], era5_v_m[:, i])   for i, a in enumerate(LIDAR_ALTS)}
    lidar_uv_by_alt = {a: (lidar_u_m[:, i], lidar_v_m[:, i]) for i, a in enumerate(LIDAR_ALTS)}

    # Data dicts for profile / clustering plots
    era5_data = {
        'wind_speed_east':  era5_u_m,
        'wind_speed_north': era5_v_m,
        'altitude': alts,
        'n_samples': n_hours,
    }
    lidar_data = {
        'wind_speed_east':  lidar_u_m,
        'wind_speed_north': lidar_v_m,
        'altitude': alts,
        'n_samples': n_hours,
    }

    # --- Plot 2: Mean wind speed comparison (all lidar altitudes) ---
    print("\n--- Plot 2: Mean wind speed comparison ---")
    plot_mean_wind_speed(era5_by_alt, lidar_by_alt, LIDAR_ALTS, n_hours,
                         savePath=_path('era5_lidar_mean_wind_speed.pdf'))

    # --- Plot 3: Weibull comparison (100 m and 200 m) ---
    print("\n--- Plot 3: Weibull comparison ---")
    plot_weibull_comparison(era5_by_alt, lidar_by_alt, WEIBULL_ROSE_ALTS, n_hours,
                            savePath=_path('era5_lidar_weibull_comparison.pdf'))

    # --- Plot 4: Wind rose comparison (100 m and 200 m) ---
    print("\n--- Plot 4: Wind rose comparison ---")
    plot_wind_rose_comparison(era5_uv_by_alt, lidar_uv_by_alt, WEIBULL_ROSE_ALTS, n_hours,
                              savePath=_path('era5_lidar_wind_rose.pdf'))

    # --- Plot 5: Mean wind speed profiles ---
    print("\n--- Plot 5: Mean wind speed profiles ---")
    plot_mean_profiles(era5_data, lidar_data, n_hours,
                       savePath=_path('era5_lidar_mean_profiles.pdf'))

    # --- Plot 6: Logarithmic profile fitting ---
    print("\n--- Plot 6: Logarithmic profile fitting ---")
    plot_log_fit_comparison(era5_data, lidar_data, n_hours,
                            savePath=_path('era5_lidar_log_fit.pdf'))

    # --- Plot 7 & 9: ERA5 and Lidar clustering (compute both before plotting) ---
    print("\n--- Plot 7: ERA5 clustering ---")
    print(f"  Clustering ERA5 with {N_CLUSTERS} clusters ...")
    era5_clust = perform_clustering_analysis(era5_data, N_CLUSTERS, ref_height=REF_HEIGHT)

    print("\n--- Plot 9: Lidar-only clustering (independent) ---")
    print(f"  Clustering lidar with {N_CLUSTERS} clusters ...")
    lidar_clust = perform_clustering_analysis(lidar_data, N_CLUSTERS, ref_height=REF_HEIGHT)

    # Shared x-axis limits across both cluster plots
    shared_xlim = (0.6, 1.6)

    # --- Plot 7 (continued): assign lidar profiles to ERA5 clusters ---
    print("\n--- Plot 7 (continued): assigning lidar → ERA5 clusters ---")
    lidar_processed = preprocess_data(lidar_data, remove_low_wind_samples=False,
                                       ref_height=REF_HEIGHT)
    lidar_labels, lidar_freq = predict_cluster(
        lidar_processed['training_data'],
        N_CLUSTERS,
        era5_clust['clusteringResults']['data_processing_pipeline'].predict,
        era5_clust['clusteringResults']['cluster_mapping'],
    )

    era5_freq = np.array(era5_clust['clusteringResults']['frequency_clusters'])
    print(f"  ERA5  cluster frequencies: {' '.join(f'{f:.1f}%' for f in era5_freq)}")
    print(f"  Lidar-on-ERA5 frequencies: {' '.join(f'{f:.1f}%' for f in lidar_freq)}")

    _, sortIdx_era5 = plot_era5_clusters(
        era5_clust, alts, refHeight=REF_HEIGHT, datasetLabel='ERA5',
        xlim=shared_xlim, savePath=_path('era5_clusters.pdf'),
    )

    plot_cluster_frequency_comparison(
        era5_freq[sortIdx_era5], lidar_freq[sortIdx_era5],
        savePath=_path('era5_lidar_cluster_frequencies.pdf'),
    )

    # --- Plot 8: Profile shape analysis (shear exponent / LLJ) ---
    print("\n--- Plot 8: Profile shape analysis (shear exponent / LLJ) ---")
    plot_profile_shape_analysis(
        era5_data, lidar_data, n_hours,
        savePath=_path('era5_lidar_profile_shape.pdf'),
    )

    # --- Plot 9 (continued): assign ERA5 profiles to lidar clusters ---
    print("\n--- Plot 9 (continued): assigning ERA5 → lidar clusters ---")
    era5_processed = preprocess_data(era5_data, remove_low_wind_samples=False,
                                      ref_height=REF_HEIGHT)
    era5_labels_on_lidar, era5_freq_on_lidar = predict_cluster(
        era5_processed['training_data'],
        N_CLUSTERS,
        lidar_clust['clusteringResults']['data_processing_pipeline'].predict,
        lidar_clust['clusteringResults']['cluster_mapping'],
    )

    lidar_clust_freq = np.array(lidar_clust['clusteringResults']['frequency_clusters'])
    print(f"  Lidar cluster frequencies:  {' '.join(f'{f:.1f}%' for f in lidar_clust_freq)}")
    print(f"  ERA5-on-lidar frequencies:  {' '.join(f'{f:.1f}%' for f in era5_freq_on_lidar)}")

    _, sortIdx_lidar = plot_era5_clusters(
        lidar_clust, alts, refHeight=REF_HEIGHT, datasetLabel='WLS7-130 lidar',
        xlim=shared_xlim, savePath=_path('lidar_clusters.pdf'),
    )

    plot_cluster_frequency_comparison(
        era5_freq_on_lidar[sortIdx_lidar], lidar_clust_freq[sortIdx_lidar],
        savePath=_path('lidar_era5_cluster_frequencies.pdf'),
    )

    if SAVE_PLOTS:
        print(f"\nDone. Plots saved to {out_dir}")


# ---------------------------------------------------------------------------
# Matplotlib styling (publication quality, LaTeX if available)
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    'font.family'         : 'serif',
    'font.size'           : 10,
    'axes.labelsize'      : 10,
    'xtick.labelsize'     : 9,
    'ytick.labelsize'     : 9,
    'legend.fontsize'     : 9,
    'axes.prop_cycle'     : mpl.cycler('color', [
        '#0072B2', '#D55E00', '#009E73',
        '#E69F00', '#CC79A7', '#56B4E9',
    ]),
    'lines.linewidth'     : 1.5,
    'axes.linewidth'      : 0.8,
    'xtick.direction'     : 'in',
    'ytick.direction'     : 'in',
    'xtick.minor.visible' : True,
    'ytick.minor.visible' : True,
    'xtick.major.size'    : 4,
    'ytick.major.size'    : 4,
    'xtick.minor.size'    : 2,
    'ytick.minor.size'    : 2,
    'xtick.major.width'   : 0.8,
    'ytick.major.width'   : 0.8,
    'xtick.minor.width'   : 0.6,
    'ytick.minor.width'   : 0.6,
    'lines.markersize'    : 4,
    'legend.frameon'      : False,
    'savefig.bbox'        : 'tight',
    'savefig.dpi'         : 300,
    **(
        {'text.usetex'        : True,
         'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
         'pgf.texsystem'      : 'pdflatex',
         'pgf.rcfonts'        : False}
        if __import__('shutil').which('latex') else
        {'text.usetex'        : False,
         'mathtext.fontset'   : 'cm'}
    ),
})


if __name__ == '__main__':
    main()
