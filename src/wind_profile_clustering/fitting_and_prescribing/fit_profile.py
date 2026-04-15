"""Fit logarithmic or power law wind speed profiles to wind data.

Wind speed is computed as the horizontal magnitude sqrt(u_east² + u_north²).
The fitted profile represents the mean vertical wind speed shape and is
exported with u_normalized equal to the fitted values (normalised to 1 at
the reference height) and v_normalized set to zero.
"""

import numpy as np
from scipy.optimize import curve_fit

KARMAN_CONSTANT = 0.41  # Von Kármán constant [-]


def _log_profile_func(z, frictVel, roughnessLength):
    """Evaluate logarithmic wind profile U(z) = (u* / kappa) * ln(z / z0).

    Args:
        z (ndarray): Heights [m], must be > 0.
        frictVel (float): Friction velocity u* [m/s].
        roughnessLength (float): Roughness length z0 [m].

    Returns:
        ndarray: Wind speed [m/s].
    """
    return (frictVel / KARMAN_CONSTANT) * np.log(z / roughnessLength)


def _power_law_func(z, uRef, alpha, zRef):
    """Evaluate power law wind profile U(z) = U_ref * (z / z_ref)^alpha.

    Args:
        z (ndarray): Heights [m].
        uRef (float): Reference wind speed [m/s].
        alpha (float): Power law exponent [-].
        zRef (float): Reference height [m].

    Returns:
        ndarray: Wind speed [m/s].
    """
    return uRef * (z / zRef) ** alpha


def _interp_at_height(data2d, heights, targetHeight):
    """Vectorised linear interpolation of a (nSamples, nAltitudes) array at one height.

    Args:
        data2d (ndarray): Shape (nSamples, nAltitudes).
        heights (ndarray): Shape (nAltitudes,), sorted ascending.
        targetHeight (float): Target height [m].

    Returns:
        ndarray: Shape (nSamples,), interpolated values.
    """
    idx = int(np.searchsorted(heights, targetHeight))
    idx = int(np.clip(idx, 1, len(heights) - 1))
    h0, h1 = heights[idx - 1], heights[idx]
    w = (targetHeight - h0) / (h1 - h0)
    return (1.0 - w) * data2d[:, idx - 1] + w * data2d[:, idx]


def fit_log_profile(heights, meanWindSpeed):
    """Fit a logarithmic profile to a mean wind speed profile.

    Fits U(z) = (u* / kappa) * ln(z / z0) using non-linear least squares.
    Only heights greater than zero are used during the fit.

    Args:
        heights (ndarray): Height levels [m].
        meanWindSpeed (ndarray): Mean wind speed at each height [m/s].

    Returns:
        dict: Fitted parameters with keys:
            - 'friction_velocity': Friction velocity u* [m/s].
            - 'roughness_length': Roughness length z0 [m].
    """
    validMask = heights > 0
    popt, _ = curve_fit(
        _log_profile_func,
        heights[validMask],
        meanWindSpeed[validMask],
        p0=[0.5, 0.01],
        bounds=([0.0, 1e-6], [20.0, 10.0]),
        maxfev=10000,
    )
    return {'friction_velocity': float(popt[0]), 'roughness_length': float(popt[1])}


def fit_power_law_profile(heights, meanWindSpeed, refHeight):
    """Fit a power law profile to a mean wind speed profile.

    Fits U(z) = U_ref * (z / z_ref)^alpha using non-linear least squares.
    U_ref is taken directly from the mean profile at refHeight and is not
    a free parameter of the fit.

    Args:
        heights (ndarray): Height levels [m].
        meanWindSpeed (ndarray): Mean wind speed at each height [m/s].
        refHeight (float): Reference height [m].

    Returns:
        dict: Fitted parameters with keys:
            - 'u_ref': Reference wind speed at refHeight [m/s].
            - 'alpha': Power law exponent [-].
            - 'ref_height': Reference height [m].
    """
    uRef = float(np.interp(refHeight, heights, meanWindSpeed))

    def _fit_func(z, alpha):
        return uRef * (z / refHeight) ** alpha

    popt, _ = curve_fit(_fit_func, heights, meanWindSpeed, p0=[0.14], bounds=([0.0], [2.0]))
    return {'u_ref': uRef, 'alpha': float(popt[0]), 'ref_height': float(refHeight)}


def fit_wind_profile(data, profileType='logarithmic', refHeight=100.0):
    """Fit a logarithmic or power law profile to wind data.

    Wind speed magnitude sqrt(u_east² + u_north²) is computed at each altitude
    and timestep. The specified profile type is fitted to the time-averaged wind
    speed profile. The result has u_normalized equal to the fitted profile
    (normalised to 1 at refHeight) and v_normalized set to zero, since
    directionality is captured by the probability matrix.

    Args:
        data (dict): Wind data with keys:
            - 'wind_speed_east': East component (nSamples, nAltitudes) [m/s].
            - 'wind_speed_north': North component (nSamples, nAltitudes) [m/s].
            - 'altitude': Height levels (nAltitudes,) [m].
        profileType (str): Profile type to fit: 'logarithmic' or 'power_law'.
            Defaults to 'logarithmic'.
        refHeight (float): Reference height for normalisation [m]. Defaults to 100.0.

    Raises:
        ValueError: If profileType is unknown or the fitted profile has zero
            wind speed at the reference height.

    Returns:
        dict: Results compatible with export_wind_profile_shapes_and_probabilities:
            - 'prl': Fitted u-component (1, nAltitudes), normalised to 1 at refHeight.
            - 'prp': Zero v-component (1, nAltitudes).
            - 'labelsFull': All-zero labels (nSamples,).
            - 'normalisationWindSpeeds': Wind speed magnitude at refHeight per sample.
            - 'windDirections': Wind direction at refHeight per sample [rad].
            - 'nSamples': Number of samples.
            - 'fitParams': Fitted profile parameters dict.
            - 'profileType': Profile type used.
    """
    u = data['wind_speed_east']
    v = data['wind_speed_north']
    heights = np.asarray(data['altitude'], dtype=float)
    nSamples = u.shape[0]

    # Compute wind speed magnitudes and mean profile
    windSpeeds = np.sqrt(u ** 2 + v ** 2)
    meanWindSpeed = np.mean(windSpeeds, axis=0)

    # Fit the chosen profile type
    if profileType == 'logarithmic':
        fitParams = fit_log_profile(heights, meanWindSpeed)
        fittedProfile = _log_profile_func(
            heights, fitParams['friction_velocity'], fitParams['roughness_length']
        )
    elif profileType == 'power_law':
        fitParams = fit_power_law_profile(heights, meanWindSpeed, refHeight)
        fittedProfile = _power_law_func(
            heights, fitParams['u_ref'], fitParams['alpha'], fitParams['ref_height']
        )
    else:
        raise ValueError(
            f"Unknown profile type: '{profileType}'. Choose 'logarithmic' or 'power_law'."
        )

    # Normalise fitted profile to 1 at reference height
    uAtRef = float(np.interp(refHeight, heights, fittedProfile))
    if uAtRef == 0.0:
        raise ValueError("Fitted profile has zero wind speed at the reference height.")
    fittedProfileNorm = fittedProfile / uAtRef

    # Prepare arrays for export — single "cluster"
    prl = fittedProfileNorm[np.newaxis, :]   # (1, nAltitudes)
    prp = np.zeros_like(prl)                 # (1, nAltitudes)
    labelsFull = np.zeros(nSamples, dtype=int)

    # Vectorised interpolation at refHeight for all samples
    normalisationWindSpeeds = _interp_at_height(windSpeeds, heights, refHeight)
    uAtRefPerSample = _interp_at_height(u, heights, refHeight)
    vAtRefPerSample = _interp_at_height(v, heights, refHeight)
    windDirections = np.arctan2(vAtRefPerSample, uAtRefPerSample)

    return {
        'prl': prl,
        'prp': prp,
        'labelsFull': labelsFull,
        'normalisationWindSpeeds': normalisationWindSpeeds,
        'windDirections': windDirections,
        'nSamples': nSamples,
        'fitParams': fitParams,
        'profileType': profileType,
    }
