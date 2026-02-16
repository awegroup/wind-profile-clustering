# Wind-profile-clustering

This repository contains the Python code for analysing vertical wind profile patterns in any dataset containing time series of wind speeds and directions for multiple altitudes. This analysis is the backbone of the [Clustering wind profile shapes to estimate airborne wind energy production](https://doi.org/10.5194/wes-5-1097-2020) paper, which has been published in Wind Energy Science.

The code has been originally developed for analysing the Dutch offshore wind atlas (DOWA) dataset. The DOWA file reading functionalities are compatible with the [time series files from 2008-2017 at 10-600 meter height at individual 2,5 km grid location](https://dataplatform.knmi.nl/catalog/datasets/index.html?x-dataset=dowa_netcdf_ts_singlepoint&x-dataset-version=1). Additionally, file reading functionalities are provided for the raw output files of the Leosphere WindCube v.2.1.8 lidar. Measurements of this machine at a location near Köln in Germany for the first three months of 2020 are provided by GWU Umwelttechnik and are under analysis by an airborne wind energy (AWE) resource consortium, which aims for developing AWE system design load case standards. A request to publish the hour-averaged measurement in this repository is pending. 

Afterwards, functionality to read and use ERA5 data has been added. ERA5 provides data on model levels rather than fixed altitude levels, requiring conversion to heights above ground. The code supports three methods for this calculation: (1) direct geopotential-based calculation using geopotential fields from ERA5, providing the most accurate time-varying altitudes; (2) hydrostatic equation calculation using temperature, humidity, and surface pressure data, also accounting for temporal variations; and (3) approximate altitude mapping using a predefined model-level-to-altitude lookup table, which is computationally efficient but assumes standard atmospheric conditions. Method 3 is used by default for simplicity, but methods 1 or 2 can be enabled by downloading the required additional ERA5 variables.

## Installing the environment and running the code

The code is tested in an Anaconda environment with Python 3.9.1 or higher. Create the environment using:

```bash
conda create --name [env_name] --file requirements.txt python=3.9.1
```

Replace `[env_name]` with a name of your choice. Activate the environment:

```bash
conda activate [env_name]
```

### Data Setup

The scripts automatically look for data in their respective folders within the `data/` directory:
- **ERA5 data**: Place NetCDF files in `data/era5/`
- **DOWA data**: Place NetCDF files in `data/dowa/`
- **FGW lidar data**: Place raw `.rtd` files or the downsampled CSV in `data/fgw_lidar/`

For DOWA data, download the [time series files from 2008-2017 at 10-600 meter height](https://dataplatform.knmi.nl/catalog/datasets/index.html?x-dataset=dowa_netcdf_ts_singlepoint&x-dataset-version=1) for your desired grid location.

### Running the Clustering Analysis

Configure the data source and clustering parameters in the script `scripts/run_and_export_clustering.py`:
- Set `DATA_SOURCE` to `'era5'`, `'fgw_lidar'`, or `'dowa'`
- Set `N_CLUSTERS` to your desired number of clusters (default: 8)
- Set `SAVE_PLOTS` to save visualizations as PDF files (default: True)

Run the script:

```bash
python scripts/run_and_export_clustering.py
```

This will perform the clustering analysis, generate visualizations, and export results to YAML format in the `results/` directory.

