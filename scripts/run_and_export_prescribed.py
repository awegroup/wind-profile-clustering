"""Run prescribed wind profile export to YAML.

This script builds a wind resource file from a user-supplied logarithmic or
power law profile. No measured wind data is required. The wind speed probability
distribution is a Weibull distribution defined by a mean wind speed and shape
factor k, with a single wind-direction bin (omnidirectional).

For 'logarithmic', supply FRICTION_VELOCITY and ROUGHNESS_LENGTH.
For 'power_law', supply ALPHA.

u_normalized contains the prescribed profile normalised to 1 at REF_HEIGHT;
v_normalized is zero for all altitudes.
"""

import sys
from pathlib import Path
import numpy as np

# Add src directory to path for imports
srcPath = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(srcPath))

from wind_profile_clustering.fitting_and_prescribing.prescribe_profile import prescribe_wind_profile
from wind_profile_clustering.export_profiles_and_probabilities_yml import (
    export_wind_profile_shapes_and_probabilities
)


def main():
    """Run prescribed wind profile export.

    User can configure:
    - PROFILE_TYPE: 'logarithmic' or 'power_law'
    - REF_HEIGHT: Reference height for profile normalisation [m]
    - ALT_MIN, ALT_MAX, ALT_STEP: Altitude grid [m]
    - MEAN_WIND_SPEED: Mean wind speed for Weibull distribution [m/s]
    - WEIBULL_K: Weibull shape factor k [-]
    - Profile parameters depending on PROFILE_TYPE (see below)
    - AWESIO_VALIDATE: Validate the exported YAML with awesio
    """
    # =============================================================================
    # USER CONFIGURATION
    # =============================================================================
    PROFILE_TYPE = 'logarithmic'  # 'logarithmic' or 'power_law'
    REF_HEIGHT = 200.0            # Reference height for normalisation [m]
    AWESIO_VALIDATE = True

    # --- Altitude grid ---
    ALT_MIN = 10.0    # Minimum altitude [m]
    ALT_MAX = 500.0   # Maximum altitude [m]
    ALT_STEP = 10.0   # Altitude step [m]

    # --- Weibull wind speed distribution ---
    MEAN_WIND_SPEED = 10.0  # Mean wind speed at reference height [m/s]
    WEIBULL_K = 2.0         # Weibull shape factor k [-]
    N_SAMPLES = 100000      # Number of synthetic samples (more = smoother distribution)

    # --- Logarithmic profile parameters (used when PROFILE_TYPE='logarithmic') ---
    FRICTION_VELOCITY = 0.4  # u* [m/s]
    ROUGHNESS_LENGTH = 0.03  # z0 [m]  (e.g. 0.0002 open sea, 0.03 open land, 0.5 suburbs)

    # --- Power law parameters (used when PROFILE_TYPE='power_law') ---
    ALPHA = 0.14  # power law exponent [-]  (e.g. 1/7 ≈ 0.143 neutral atmosphere)

    # --- Output metadata ---
    NAME = 'Prescribed Wind Profile'
    DESCRIPTION = 'Wind resource file with a prescribed analytical wind profile'
    # =============================================================================

    altitudes = np.arange(ALT_MIN, ALT_MAX + ALT_STEP * 0.5, ALT_STEP)
    print(f"Altitude grid: {ALT_MIN:.0f} – {ALT_MAX:.0f} m in {ALT_STEP:.0f} m steps "
          f"({len(altitudes)} levels)")

    # Build the prescribed profile + Weibull distribution
    print(f"\nBuilding prescribed {PROFILE_TYPE} profile...")

    if PROFILE_TYPE == 'logarithmic':
        result = prescribe_wind_profile(
            altitudes,
            profileType='logarithmic',
            refHeight=REF_HEIGHT,
            meanWindSpeed=MEAN_WIND_SPEED,
            weibullK=WEIBULL_K,
            nSamples=N_SAMPLES,
            frictionVelocity=FRICTION_VELOCITY,
            roughnessLength=ROUGHNESS_LENGTH,
        )
        paramStr = (
            f"friction velocity u* = {FRICTION_VELOCITY} m/s, "
            f"roughness length z0 = {ROUGHNESS_LENGTH} m"
        )
        profileFormula = "U(z) = (u*/kappa) * ln(z/z0)"

    elif PROFILE_TYPE == 'power_law':
        result = prescribe_wind_profile(
            altitudes,
            profileType='power_law',
            refHeight=REF_HEIGHT,
            meanWindSpeed=MEAN_WIND_SPEED,
            weibullK=WEIBULL_K,
            nSamples=N_SAMPLES,
            alpha=ALPHA,
        )
        paramStr = f"exponent alpha = {ALPHA}"
        profileFormula = "U(z) = U_ref * (z/z_ref)**alpha"

    else:
        raise ValueError(
            f"Unknown profile type: {PROFILE_TYPE}. Choose 'logarithmic' or 'power_law'."
        )

    print(f"Profile parameters: {result['profileParams']}")
    print(f"Weibull parameters: {result['weibullParams']}")

    note = (
        f"Profile shape prescribed analytically using a {PROFILE_TYPE} profile "
        f"({profileFormula}) with {paramStr}. "
        f"No measured wind data was used. "
        f"The wind speed probability distribution is a Weibull distribution with "
        f"mean wind speed {MEAN_WIND_SPEED} m/s and shape factor k = {WEIBULL_K} "
        f"(Weibull scale parameter lambda = {result['weibullParams']['lambda']:.4f} m/s). "
        f"u_normalized contains the prescribed profile normalised to 1 at "
        f"{REF_HEIGHT:.0f} m; v_normalized is zero for all altitudes. "
        f"The probability matrix has a single wind-direction bin (omnidirectional)."
    )

    metadata = {
        'name': NAME,
        'description': DESCRIPTION,
        'note': note,
        'data_source': 'prescribed_analytical',
    }

    # Export to YAML — use windDirectionBinWidth=360 for a single omnidirectional bin
    print("\nExporting results to YAML...")
    outputFile = f'results/wind_profile_prescribed_{PROFILE_TYPE}.yml'

    probMatrix = export_wind_profile_shapes_and_probabilities(
        altitudes,
        result['prl'],
        result['prp'],
        result['labelsFull'],
        result['normalisationWindSpeeds'],
        result['windDirections'],
        result['nSamples'],
        1,  # single prescribed profile
        outputFile,
        metadata=metadata,
        refHeight=REF_HEIGHT,
        windDirectionBinWidth=360,
        validate=AWESIO_VALIDATE,
    )

    print(f"\nExported prescribed profile to: {outputFile}")
    print(f"Probability matrix shape: {probMatrix.shape}")
    print(f"Total probability sum: {np.sum(probMatrix):.2f}%")



if __name__ == '__main__':
    main()
