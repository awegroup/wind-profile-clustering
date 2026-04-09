"""Prescribe a logarithmic or power law wind speed profile analytically.

No wind measurement data is required. The wind speed probability distribution
is a 1-D Weibull distribution (no wind direction) generated from a user-supplied
mean wind speed and Weibull shape factor k.

The exported profile has:
- u_normalized: the prescribed profile shape normalised to 1 at the reference height.
- v_normalized: zero for all altitudes (directionality is not represented).

The probability matrix has a single wind-direction bin (0–360°) so that it
represents a purely scalar wind speed distribution.
"""

import numpy as np
from scipy.special import gamma

from wind_profile_clustering.fitting_and_prescribing.fit_profile import (
    _log_profile_func,
    _power_law_func,
    KARMAN_CONSTANT,
)


def prescribe_wind_profile(heights, profileType='logarithmic', refHeight=100.0,
                           meanWindSpeed=10.0, weibullK=2.0, nSamples=100000,
                           alpha=None, frictionVelocity=None, roughnessLength=None):
    """Build a wind resource file from a prescribed profile and a Weibull wind speed distribution.

    The profile shape is computed analytically from the supplied parameters.
    No measured wind data is used. The wind speed probability distribution is
    drawn from a Weibull distribution characterised by meanWindSpeed and weibullK.
    All samples are assigned a wind direction of zero so that, when the export
    function is called with windDirectionBinWidth=360, the result is a purely
    1-D wind speed distribution (single direction bin covering 0–360°).

    u_normalized contains the prescribed profile normalised to 1 at refHeight;
    v_normalized is zero for all altitudes.

    Args:
        heights (ndarray): Height levels [m].
        profileType (str): 'logarithmic' or 'power_law'. Defaults to 'logarithmic'.
        refHeight (float): Reference height for normalisation [m]. Defaults to 100.0.
        meanWindSpeed (float): Mean wind speed at the reference height [m/s].
            Defaults to 10.0.
        weibullK (float): Weibull shape factor k [-]. Defaults to 2.0.
        nSamples (int): Number of synthetic Weibull samples to generate. A larger
            value gives a smoother probability matrix. Defaults to 100000.
        alpha (float): Power law exponent [-]. Required when profileType='power_law'.
        frictionVelocity (float): Friction velocity u* [m/s]. Required when
            profileType='logarithmic'.
        roughnessLength (float): Roughness length z0 [m]. Required when
            profileType='logarithmic'.

    Raises:
        ValueError: If required parameters for the chosen profile type are missing,
            or if the profile evaluates to zero at the reference height.

    Returns:
        dict: Results compatible with export_wind_profile_shapes_and_probabilities
            (pass windDirectionBinWidth=360 when calling that function):
            - 'prl': Prescribed u-component (1, nAltitudes), normalised to 1 at refHeight.
            - 'prp': Zero v-component (1, nAltitudes).
            - 'labelsFull': All-zero labels (nSamples,).
            - 'normalisationWindSpeeds': Weibull-sampled wind speeds at refHeight (nSamples,).
            - 'windDirections': Zero-valued directions (nSamples,) [rad].
            - 'nSamples': Number of synthetic samples.
            - 'profileParams': Parameters used to construct the profile.
            - 'weibullParams': Weibull distribution parameters.
            - 'profileType': Profile type used.
    """
    heights = np.asarray(heights, dtype=float)

    # --- Compute analytical profile shape ---
    if profileType == 'logarithmic':
        if frictionVelocity is None or roughnessLength is None:
            raise ValueError(
                "frictionVelocity and roughnessLength must be provided for 'logarithmic' profile."
            )
        validMask = heights > 0
        profile = np.where(
            validMask,
            _log_profile_func(np.where(validMask, heights, 1.0), frictionVelocity, roughnessLength),
            0.0,
        )
        profileParams = {
            'friction_velocity': float(frictionVelocity),
            'roughness_length': float(roughnessLength),
        }

    elif profileType == 'power_law':
        if alpha is None:
            raise ValueError("alpha must be provided for 'power_law' profile.")
        profile = _power_law_func(heights, 1.0, alpha, refHeight)
        profileParams = {'alpha': float(alpha), 'ref_height': float(refHeight)}

    else:
        raise ValueError(
            f"Unknown profile type: '{profileType}'. Choose 'logarithmic' or 'power_law'."
        )

    # Normalise profile to 1 at reference height
    uAtRef = float(np.interp(refHeight, heights, profile))
    if uAtRef == 0.0:
        raise ValueError("Prescribed profile has zero wind speed at the reference height.")
    profileNorm = profile / uAtRef

    # --- Generate Weibull wind speed distribution ---
    # Weibull scale parameter lambda from the mean: mean = lambda * Gamma(1 + 1/k)
    lambdaParam = meanWindSpeed / gamma(1.0 + 1.0 / weibullK)
    weibullParams = {
        'mean_wind_speed': float(meanWindSpeed),
        'k': float(weibullK),
        'lambda': float(lambdaParam),
    }

    rng = np.random.default_rng(0)
    normalisationWindSpeeds = lambdaParam * rng.weibull(weibullK, nSamples)

    # All directions are zero → single bin when windDirectionBinWidth=360
    windDirections = np.zeros(nSamples)
    labelsFull = np.zeros(nSamples, dtype=int)

    prl = profileNorm[np.newaxis, :]  # (1, nAltitudes)
    prp = np.zeros_like(prl)          # (1, nAltitudes)

    return {
        'prl': prl,
        'prp': prp,
        'labelsFull': labelsFull,
        'normalisationWindSpeeds': normalisationWindSpeeds,
        'windDirections': windDirections,
        'nSamples': nSamples,
        'profileParams': profileParams,
        'weibullParams': weibullParams,
        'profileType': profileType,
    }
