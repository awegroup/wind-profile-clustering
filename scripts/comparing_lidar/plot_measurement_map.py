"""Map of the ERA5 grid and WLS7-130 lidar location at Bangor Erris, Ireland.

Reads one ERA5 surface file to extract the 0.25-degree lat/lon grid,
identifies the four surrounding ERA5 grid cells and plots them alongside the
lidar site on an OpenStreetMap basemap.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as cx

srcPath = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(srcPath))

# =============================================================================
# CONFIGURATION
# =============================================================================
LIDAR_LAT   = 54.1254
LIDAR_LON   = -9.7801
ERA5_DATA_DIR = 'data/era5'

# Local map extent [lon_min, lon_max, lat_min, lat_max]
MAP_EXTENT  = (-10.5, -8.5, 53.0, 55.0)

SAVE_PLOT   = True
# =============================================================================


def _find_surrounding_points(lats, lons, targetLat, targetLon):
    """Return the four ERA5 grid points enclosing the target location.

    Args:
        lats (ndarray): 1-D array of latitude grid values.
        lons (ndarray): 1-D array of longitude grid values.
        targetLat (float): Target latitude in degrees.
        targetLon (float): Target longitude in degrees.

    Returns:
        tuple: (surr_lats, surr_lons) each a list of four values ordered
            [SW, SE, NW, NE].
    """
    below = lats[lats <= targetLat]
    above = lats[lats >= targetLat]
    latS  = float(below.max()) if below.size else float(lats.min())
    latN  = float(above.min()) if above.size else float(lats.max())

    left  = lons[lons <= targetLon]
    right = lons[lons >= targetLon]
    lonW  = float(left.max())  if left.size  else float(lons.min())
    lonE  = float(right.min()) if right.size else float(lons.max())

    surr_lats = [latS, latS, latN, latN]
    surr_lons = [lonW, lonE, lonW, lonE]
    return surr_lats, surr_lons


def plot_measurement_map(savePath=None):
    """Plot an OpenStreetMap basemap with ERA5 grid and lidar location.

    Reads the ERA5 lat/lon grid from the first available surface NetCDF file,
    identifies the four grid points surrounding the lidar site, and overlays
    all local ERA5 grid points, the surrounding cell, and the lidar location.

    Args:
        savePath (str or Path, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    # --- Load ERA5 grid ---
    sfcFiles = sorted(Path(ERA5_DATA_DIR).glob('sfc_*.netcdf'))
    if not sfcFiles:
        raise FileNotFoundError(f"No ERA5 surface files found in {ERA5_DATA_DIR}")

    ds   = xr.open_dataset(sfcFiles[0])
    lats = np.sort(ds['latitude'].values)    # ascending
    lons = np.sort(ds['longitude'].values)   # ascending
    ds.close()

    # Restrict to map extent
    latMask = (lats >= MAP_EXTENT[2]) & (lats <= MAP_EXTENT[3])
    lonMask = (lons >= MAP_EXTENT[0]) & (lons <= MAP_EXTENT[1])
    localLats = lats[latMask]
    localLons = lons[lonMask]
    gridLon, gridLat = np.meshgrid(localLons, localLats)

    # Four surrounding ERA5 grid points
    surrLats, surrLons = _find_surrounding_points(lats, lons, LIDAR_LAT, LIDAR_LON)

    # Cell boundary polygon (closed)
    latS, latN = min(surrLats), max(surrLats)
    lonW, lonE = min(surrLons), max(surrLons)
    cellLons   = [lonW, lonE, lonE, lonW, lonW]
    cellLats   = [latS, latS, latN, latN, latS]

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(6, 4))

    # All local ERA5 grid points
    ax.scatter(
        gridLon.ravel(), gridLat.ravel(),
        s=18, marker='+', linewidths=0.8,
        color='#0072B2', alpha=0.55, zorder=3,
        label='ERA5 grid points (0.25°)',
    )

    # Surrounding cell boundary
    ax.plot(
        cellLons, cellLats,
        '--', color='#0072B2', lw=1.2, alpha=0.9, zorder=4,
    )

    # Four surrounding grid points (highlighted)
    ax.scatter(
        surrLons, surrLats,
        s=70, marker='+', linewidths=1.8,
        color='#0072B2', zorder=5,
        label='Surrounding ERA5 grid points',
    )

    # Lidar location
    ax.scatter(
        [LIDAR_LON], [LIDAR_LAT],
        s=130, marker='*',
        color='#D55E00', edgecolors='white', linewidths=0.5,
        zorder=6,
        label=f'WLS7-130 lidar ({LIDAR_LAT:.4f}°N, {abs(LIDAR_LON):.4f}°W)',
    )

    # OpenStreetMap basemap (contextily reprojects tiles to EPSG:4326)
    cx.add_basemap(
        ax,
        crs='EPSG:4326',
        source=cx.providers.OpenStreetMap.Mapnik,
        zoom=7,
        alpha=0.7,
        zorder=1,
    )

    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.25, zorder=2)
    fig.tight_layout()

    if savePath:
        Path(savePath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savePath)
        print(f"Saved: {savePath}")
        plt.close(fig)
    return fig


def main():
    """Entry point: generate and save the measurement map."""
    outPath = None
    if SAVE_PLOT:
        outPath = Path(__file__).parent.parent / 'results' / 'plots' / 'era5_lidar_map.pdf'
    plot_measurement_map(savePath=outPath)


# ---------------------------------------------------------------------------
# Matplotlib styling (matches era5_lidar_analysis.py)
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    'font.family'         : 'serif',
    'font.size'           : 10,
    'axes.labelsize'      : 10,
    'xtick.labelsize'     : 9,
    'ytick.labelsize'     : 9,
    'legend.fontsize'     : 9,
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
