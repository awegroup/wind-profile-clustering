# Wind-profile-clustering

This repository contains the Python code for analysing vertical wind profile patterns in any dataset containing time series of wind speeds and directions for multiple altitudes. This analysis is the backbone of the [Clustering wind profile shapes to estimate airborne wind energy production](https://doi.org/10.5194/wes-5-1097-2020) paper, which has been published in Wind Energy Science.

The code has been originally developed for analysing the Dutch offshore wind atlas (DOWA) dataset. The DOWA file reading functionalities are compatible with the [time series files from 2008-2017 at 10-600 meter height at individual 2,5 km grid location](https://dataplatform.knmi.nl/catalog/datasets/index.html?x-dataset=dowa_netcdf_ts_singlepoint&x-dataset-version=1). Additionally, file reading functionalities are provided for the raw output files of the Leosphere WindCube v.2.1.8 lidar. Measurements of this machine at a location near Köln in Germany for the first three months of 2020 are provided by GWU Umwelttechnik and are under analysis by an airborne wind energy (AWE) resource consortium, which aims for developing AWE system design load case standards. A request to publish the hour-averaged measurement in this repository is pending. 

Afterwards, functionality to read and use ERA5 data has been added. ERA5 provides data on model levels rather than fixed altitude levels, requiring conversion to heights above ground. The code supports two methods for this calculation: (1) hydrostatic equation calculation using temperature, humidity, and surface pressure data, accounting for temporal variations; and (2) approximate altitude mapping using a predefined model-level-to-altitude lookup table, which is computationally efficient but assumes standard atmospheric conditions. Method 2 is used by default for simplicity, but method 1 can be enabled by downloading the required additional ERA5 variables.

All results are exported in the [awesIO](https://github.com/awegroup/awesIO) wind resource format, a standardised YAML format for wind resource data used in airborne wind energy research. In addition to the clustering workflow, the repository also provides functionality to fit logarithmic or power law profiles to wind data, and to prescribe analytical profiles entirely without measured data.

## Installing the environment and running the code

The code is tested with Python 3.9 or higher. 

### Installation with Conda (Recommended)

Create and activate a new conda environment:

```bash
conda create --name wind_clustering python=3.9
conda activate wind_clustering
```

Install the package and its dependencies:

```bash
pip install -e .
```

### Installation with pip only

If you're not using conda, install directly with pip:

```bash
pip install -e .
```

The `-e` flag installs the package in editable mode, allowing you to modify the code without reinstalling.

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

This will perform the clustering analysis, generate visualizations, and export results to the awesIO wind resource YAML format in the `results/` directory.

### Running the Profile Fitting

Configure the data source and fitting parameters in `scripts/run_and_export_fitting.py`:
- Set `DATA_SOURCE` to `'era5'`, `'fgw_lidar'`, or `'dowa'`
- Set `PROFILE_TYPE` to `'logarithmic'` or `'power_law'`
- Set `REF_HEIGHT` to the reference height for profile normalisation (default: 200 m)

Run the script:

```bash
python scripts/run_and_export_fitting.py
```

This fits the chosen profile type to the time-averaged wind speed magnitude from the selected dataset and exports the result to the awesIO wind resource YAML format in the `results/` directory. The wind speed magnitude is computed as $\sqrt{u_{east}^2 + u_{north}^2}$; `u_normalized` contains the fitted profile normalised to 1 at the reference height and `v_normalized` is zero.

### Running the Profile Prescribing

Configure the parameters in `scripts/run_and_export_prescribed.py`:
- Set `PROFILE_TYPE` to `'logarithmic'` or `'power_law'`
- Set `REF_HEIGHT`, `ALT_MIN`, `ALT_MAX`, and `ALT_STEP` for the altitude grid
- Set `MEAN_WIND_SPEED` and `WEIBULL_K` to define the Weibull wind speed distribution
- For `'logarithmic'`: set `FRICTION_VELOCITY` and `ROUGHNESS_LENGTH`
- For `'power_law'`: set `ALPHA`

Run the script:

```bash
python scripts/run_and_export_prescribed.py
```

This builds a wind resource file entirely from prescribed parameters — no measured wind data is required. The profile shape is computed analytically and the wind speed probability distribution is a Weibull distribution. Results are exported to the awesIO wind resource YAML format in the `results/` directory.

