"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Physics calculations for photon-matter interactions in germanium
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numba
import numpy as np
from scipy.constants import physical_constants
from scipy.interpolate import PchipInterpolator, interp1d  # CubicSpline,

# from numba import njit
# from functools import partial

# from greto.utils import log_interp

# %% Constants
RHO_GE = 5.323  # Density of Germanium [g/cm3] (NIST)
Z_GE = 32  # Atomic Number of Germanium [p+/atom]
A_GE = 74  # Atomic Mass of Germanium [g/mol]
Z_A_GE = 0.44071  # Ratio of Z/A of Germanium [p+/(p+ + n)] (NIST)
I_GE = 350.0  # Mean excitation energy of Germanium [eV] (NIST)
# Avogadro's number 6.02214076e+23 [atoms/mol]
N_AV = physical_constants["Avogadro constant"][0]
# Classical electron Radius 2.8179403262e-13 [cm]
R_0 = physical_constants["classical electron radius"][0] * 100
# electron mass 0.51099895 [MeV]
MEC2 = physical_constants["electron mass energy equivalent in MeV"][0]
# fine structure constant 1/137 ~ 0.0072973525693
ALPHA = physical_constants["fine-structure constant"][0]
# Threshold for pair production on an atomic nucleus [eV] (NIST)
THRESHOLD_PAIR_ATOM = 1.022007e06
# Threshold for pair production on an electron [eV] (NIST)
THRESHOLD_PAIR_ELECTRON = 2.044014e06
# barns per square cm [barns/cm^2]
BARNS_PER_SQCM = 1e24
# K edge in Germanium [eV] (NIST)
K_GE = 11103.1
# 4.331872333172973e+22 [(atoms/mol)*(g/cm^3)*(mol/g) = atoms/cm^3]
# conversion from cross section (cm^2/atom) to linear attenuation (1/cm)
RANGE_PROCESS = (N_AV * RHO_GE) / A_GE

# %% Compton Scattering Formula: Compton Edge
"""
There is an upper bound on deposited energy (a lower bound on continuing energy)
imposed by the Compton Edge.
"""


@numba.njit
def compton_edge_incoming(
    E_out: np.ndarray[float] | float,
) -> np.ndarray[float] | float:
    """
    Incoming energy assuming a back-scatter given outgoing energy

    Args:
        - E_out: energy leaving a Compton Scattering interaction

    Returns:
        - Incoming energy was at least...

    Outbound energy E_out is minimal and the energy deposit maximal.

    Find the energy in, E_in, assuming a maximal energy deposit.
    Assume that E_out is minimum energy coming out of a scatter (deposited the
    maximum; complete back-scatter) and compute the incoming energy that deposited it.

    Returns the minimum incoming energy given the outgoing energy (MeV) (smaller
    incoming energy implies larger than physical deposit)

    E_out = E_in / (1 + (E_in / MEC2) (1 - cos theta) )

    Solve for E_in:
        * E_out = E_in / (1 + (1 - cos theta) (E_in / MEC2))
        * E_out + (1 - cos theta) E_out/MEC2 (E_in) = E_in
        * E_out = (1 - (1 - cos theta) E_out/MEC2) E_in
        * E_out / (1 - (1 - cos theta) E_out/MEC2) = E_in
    """
    return E_out / (1 - 2 * E_out / MEC2)


@numba.njit
def compton_edge_incoming_diff(
    E_out: np.ndarray[float] | float,
) -> np.ndarray[float] | float:
    """
    Deposited energy assuming a back-scatter given outgoing energy

    What is the energy deposit that still allows for E_out assuming a
    back-scatter?

    Find the energy deposited (E_in - E_out), assuming a maximal energy deposit.
    """
    # return compton_edge_outgoing(E_out) - E_out
    return E_out * (-1 + 1 / (1 - 2 * E_out / MEC2))


@numba.njit
def compton_edge_outgoing(E_in: np.ndarray[float] | float) -> np.ndarray[float] | float:
    """
    Outgoing energy assuming a back-scatter given incoming energy

    Args:
        - E_in: energy entering a Compton Scattering interaction

    Returns:
        - Outgoing energy was at least...

    What is the smallest energy that could come out of a scatter with incoming
    energy E_in (maximal energy deposit)?

    Returns the minimum un-deposited energy given the incoming energy (MeV)

    E_out = E_in / (1 + (E_in / MEC2) (1 - cos theta) )

    Maximum deposit is a back-scatter where cos theta = -1
    """
    return E_in / (1 + 2 * E_in / MEC2)


@numba.njit
def compton_edge_outgoing_diff(
    E_in: np.ndarray[float] | float,
) -> np.ndarray[float] | float:
    """
    Deposited energy assuming a back-scatter given incoming energy

    Args:
        - E_in: energy entering a Compton Scattering interaction

    Returns:
        - Deposited energy was at most...

    What is the largest energy deposit that could happen with incoming energy
    E_in?
    """
    # return E_in - compton_edge_outgoing(E_in)
    return E_in * (1 - 1 / (1 + 2 * E_in / MEC2))


# %% Compton Scattering Formula: Cosine
@numba.njit
def njit_cos_theor(
    E_imo: float | np.ndarray[float],
    E_i: float | np.ndarray[float],
) -> np.ndarray[float]:
    """
    Compton scattering formula. Compute cosine of angle based on starting and
    ending energy after an interaction.

    Args:
        - E_imo: The energy prior to the interaction
        - E_i: The energy after the interaction
    """
    return 1 - MEC2 * (1 / E_i - 1 / E_imo)


@numba.njit
def cos_theor_sequence(energies: np.ndarray) -> np.ndarray:
    """
    Compton scattering formula. Compute cosine of angle based on starting and
    ending energy after an interaction.

    Args:
        - energies: sequence of energies [MeV]
    """
    # return 1 - MEC2 * np.diff(1 / energies)
    out = np.zeros((energies.shape[0]-1,))  # for loop is faster when compiled
    for i in range(out.shape[0]):
        out[i] = 1 - MEC2 * (1/energies[i+1] - 1/energies[i])
    return out


def cos_theor(
    E_imo: float | np.ndarray[float],
    E_i: float | np.ndarray[float],
    penalty: Optional[float] = None,
    penalty_slack: float = 0.1,
    compton_relief: float = 1.0e-10,
) -> np.ndarray[float]:
    """
    Compton scattering formula. Compute cosine of angle based on starting and
    ending energy after an interaction.

    Args:
        - E_imo: The energy prior to the interaction
        - E_i: The energy after the interaction
        - penalty: The penalty for having an unrealistic
            energy drop (cosine is < -1) violating the Compton edge
        - penalty_slack: Slack in the cosine value (if within slack of -1, add
          in compton_relief)
        - compton_relief: added if within the slack value of -1
    """
    if isinstance(E_imo, float) and penalty is not None:
        E_imo = np.array(E_imo)
        E_i = np.array(E_i)
    cos_theta = 1 - MEC2 * (1 / E_i - 1 / E_imo)
    if penalty is not None:
        cos_theta[cos_theta < -1 - penalty_slack] = penalty
        cos_theta[np.logical_and(cos_theta < -1, cos_theta > -1 - penalty_slack)] = (
            -1.0 + compton_relief
        )
    return cos_theta


@numba.njit
def compton_penalty_ell1_single(cosine: float) -> float:
    """Returns how much smaller than -1 the cosine value is or 0.0"""
    return max(0, -1.0 - cosine)


@numba.njit
def compton_penalty_ell1(cosines: np.ndarray[float]):
    """
    Return how much more negative the theoretical cosine is compared to -1

    Args:
        - cosines: theoretical cosine values

    Returns:
        - -1.0 - cosine if cosine < -1, 0.0 otherwise
    """
    # return np.clip(-cosines - 1, 0.0, np.inf)  # slightly slower but identical behavior
    # return np.where(cosines < -1, -1 - cosines, 0.0)
    # return (cosines < -1) * (-1 - cosines)  # slightly faster than where
    # return -1 - np.minimum(cosines, -1.0)  # almost 2x faster than clip
    # return np.maximum(0, -1 - cosines)
    out = np.zeros(cosines.shape)  # for loop is even faster
    for i in range(len(cosines)):
        if cosines[i] < -1:
            out[i] = -1 - cosines[i]
    return out


@numba.njit
def compton_penalty_single(cosine: float) -> float:
    """Indicator if a penalty for non-physical scattering should be applied"""
    if cosine < -1:
        return 1.0
    return 0.0


@numba.njit
def compton_penalty(cosines: np.ndarray[float]):
    """
    Return if the theoretical cosine is less than -1

    Args:
        - cosines: theoretical cosine values

    Returns:
        - 1.0 if cosine < -1, 0.0 otherwise
    """
    # return np.where(cosines < -1, 1.0, 0.0)
    # return (cosines < -1).astype(float)  # ~2x faster than where
    out = np.zeros(cosines.shape)  # for loop is even faster
    for i in range(len(cosines)):
        if cosines[i] < -1:
            out[i] = 1.0
    return out


@numba.njit
def cos_theor_sigma(
    E_imo: float | np.ndarray[float],
    E_i: float | np.ndarray[float],
    Nmi: Optional[int] = None,
    eres: float = 1e-3,
) -> np.ndarray[float]:
    """
    Compton scattering formula error. Compute cosine of angle based on starting
    and ending energy after an interaction.

    Args:
        - E_imo: The energy prior to the interaction
        - E_i: The energy after the interaction
        - Nmi: The number of interactions in the cluster
    """
    if isinstance(E_imo, float):
        E_imo = np.array(E_imo)
        E_i = np.array(E_i)
    if Nmi is None:
        nmi = np.arange(len(E_imo), 0, -1)
    else:
        Nmi = max(Nmi, len(E_imo))
        nmi = np.arange(Nmi, Nmi - len(E_imo), -1)
    sigma_squared = 1 / E_imo**4 + nmi * (1 / E_i**2 - 1 / E_imo**2)
    return MEC2 * eres * np.sqrt(sigma_squared)


# def cos_theor_err(
#     E_imo: float | np.ndarray[float],
#     E_i: float | np.ndarray[float],
#     Nmi: Optional[int] = None,
#     eres: float = 1e-3,
# ) -> np.ndarray[float]:
#     """
#     Compton scattering formula. Compute the error of computing cosine of angle
#     based on starting and ending energy after an interaction.

#     Args:
#         E_imo: The energy prior to the interaction
#         E_i: The energy after the interaction
#         Nmi: The number of interactions in the entire cluster
#         penalty: The penalty for having an unrealistic
#             energy drop (cosine is < -1) violating the Compton edge
#         penalty_slack: Slack in the cosine value
#     """
#     if isinstance(E_imo, float):
#         E_imo = np.array(E_imo)
#         E_i = np.array(E_i)
#     cos_theta = cos_theor(E_imo, E_i, penalty=None)
#     if Nmi is None:
#         return np.where(cos_theta > -1, MEC2 * eres / E_imo**2, 1)
#     return np.where(cos_theta > -1,
#         MEC2 * eres * np.sqrt( 1 / E_imo**4 + np.arange(Nmi, Nmi - len(E_imo), -1) * (1 / E_i**2 - 1 / E_imo**2) ** 2),
#         1,
#     )

@numba.njit
def theta_theor_single(
    E_imo: float,
    E_i: float,
    penalty: Optional[float] = None,
) -> float:
    """
    The arccosine of cos_theor.

    Args:
        - E_imo: The energy prior to the interaction
        - E_i: The energy after the interaction
        - penalty: Compton Edge violation penalty value
    """
    # c_theta = np.arccos(njit_cos_theor(E_imo, E_i))
    c_theta = njit_cos_theor(E_imo, E_i)
    if penalty is not None:
        if c_theta < -1:
            return penalty
    return np.arccos(c_theta)

@numba.njit
def theta_theor(
    E_imo: np.ndarray[float],
    E_i: np.ndarray[float],
    penalty: Optional[float] = None,
) -> float:
    """
    The arccosine of cos_theor.

    Args:
        - E_imo: The energy prior to the interaction
        - E_i: The energy after the interaction
        - penalty: Compton Edge violation penalty value

    >>> phys.theta_theor(
            np.array([4.0,3.0,2.0,1.0,4.0]),
            np.array([3.5,2.5,1.5,0.5,0.1]),
            -12,
        )
    array([  0.19134129,   0.26177011,   0.41570088,   1.05985215, -12.        ])
    """
    # c_theta = np.where(c_theta < -1, penalty, np.arccos(c_theta))
    # penalty_inds = c_theta < -1
    # # Assign penalty to values that violate physics
    # c_theta[penalty_inds] = penalty
    # # Take the arccos of values that are not the penalty value
    # c_theta[~penalty_inds] = np.arccos(c_theta[~penalty_inds])

    c_theta = np.zeros((E_imo.shape[0],))
    for i in numba.prange(c_theta.shape[0]):
        if E_imo[i] > 0 and E_i[i] > 0 and E_imo[i] > E_i[i]:
            c_theta[i] = theta_theor_single(E_imo[i], E_i[i], penalty)
        elif penalty is not None:
            c_theta[i] = penalty
    return c_theta


@numba.njit
def theta_theor_single(
    E_imo: float,
    E_i: float,
    penalty: Optional[float] = None,
) -> float:
    """
    Computes theoretical scattering angle using two floats
    """
    c_theta = njit_cos_theor(E_imo, E_i)
    if penalty is not None and c_theta < -1:
        return penalty
    return np.arccos(c_theta)


# %% Compton Scattering Formula: Outbound energy
@numba.njit
def outgoing_energy_csf(
    E_imo: float | np.ndarray[float], one_minus_cosines: float | np.ndarray[float]
) -> float | np.ndarray[float]:
    """
    Compton scattering formula for outgoing energy

    Args:
        - E_imo: The energy prior to the interaction
        - one_minus_cosines: 1 - cosine of scattering angle
    """
    return E_imo / (1 + (E_imo / MEC2) * one_minus_cosines)

@numba.njit
def outgoing_energy_csf_sigma(
    E_imo: np.ndarray[float],
    one_minus_cosines: np.ndarray[float],
    cosine_error: np.ndarray[float],
    cumulative_energy_error: np.ndarray[float],
) -> np.ndarray[float]:
    """
    Standard error for Compton scattering formula for outgoing energy

    Args:
        - E_imo: The energy prior to the interaction
        - one_minus_cosines: 1 - cosine of scattering angle
        - cosine_error: Error due to computing the cosines
        - cumulative_energy_error: Error due to accumulating energies
          (cumulative sum of interaction energies)
    """
    E_geo = outgoing_energy_csf(E_imo, one_minus_cosines)
    return np.sqrt(
        (cosine_error * E_geo**2 / MEC2) ** 2
        + (cumulative_energy_error * (E_geo / E_imo) ** 2) ** 2
    )


# %% Compton Scattering Formula: Inbound energy
@numba.njit
def incoming_energy_csf(
    E_i: np.ndarray[float], one_minus_cosines: np.ndarray[float]
) -> np.ndarray[float]:
    """Compton scattering formula for incoming energy"""
    return E_i / (1 - (E_i / MEC2) * one_minus_cosines)


def incoming_energy_csf_sigma(
    E_i: np.ndarray[float],
    one_minus_cosines: np.ndarray[float],
    cosine_error: np.ndarray[float],
    cumulative_energy_error: np.ndarray[float],
) -> np.ndarray[float]:
    """Standard error of Compton scattering formula for incoming energy"""
    raise NotImplementedError


# %% Compton Scattering Formula: Local inbound energy (TANGO)
@numba.njit
def tango_incoming_estimate(
    e: np.ndarray[float],
    one_minus_cosines: np.ndarray[float],
    fill_value:float = 123456789.0,  # TODO - get rid of arbitrary number
) -> np.ndarray[float]:
    """
    Estimated incoming energy from local energy and the angle of scattering
    """
    out = np.zeros(one_minus_cosines.shape)
    for i in range(one_minus_cosines.shape[0]):
        for j in range(one_minus_cosines.shape[1]):
            for k in range(one_minus_cosines.shape[2]):
                if one_minus_cosines[i,j,k] <= 0.0:
                    out[i,j,k] = fill_value
                else:
                    out[i,j,k] = 0.5 * e[j] + np.sqrt(e[j]**2 / 4 + e[j] * MEC2 / one_minus_cosines[i,j,k])
    # one_minus_cosines[one_minus_cosines <= 0.0] = 10
    return out

@numba.njit
def partial_tango_incoming_derivatives_d_de_single(e, omc):
    return 0.5 + 0.5 * (1 / np.sqrt(e**2 / 4 + e*MEC2/omc)) * (e/2 + MEC2/omc)

@numba.njit
def partial_tango_incoming_derivatives_d_d_cos_single(e, omc):
    return 0.5 * (1 / np.sqrt(e**2 / 4 + e * MEC2 / omc)) * (e * MEC2) / (omc**2)

@numba.njit
def partial_tango_incoming_derivatives(
    e: np.ndarray[float],
    o_m_cos_ijk: np.ndarray[float],
    fill_value: float = 123456789.0,  # TODO - get rid of arbitrary number
) -> np.ndarray[float]:
    """
    Partial derivatives of estimated incoming energy using local information
    """

    out_1 = np.zeros(o_m_cos_ijk.shape)
    out_2 = np.zeros(o_m_cos_ijk.shape)
    for i in range(out_1.shape[0]):
        for j in range(out_1.shape[1]):
            for k in range(out_1.shape[2]):
                if o_m_cos_ijk[i,j,k] <= 0.0 or e[j] <= 0.0:
                    out_1[i,j,k] = fill_value
                    out_2[i,j,k] = fill_value
                else:
                    # out_1[i,j,k] = 0.5 + 0.5 * (1 / np.sqrt(e[j]**2 / 4 + e[j] * MEC2 / o_m_cos_ijk[i,j,k])) * (e[j] / 2 + MEC2 / o_m_cos_ijk[i,j,k])
                    # out_2[i,j,k] = 0.5 * (1 / np.sqrt(e[j]**2 / 4 + e[j] * MEC2 / o_m_cos_ijk[i,j,k])) * (e[j] * MEC2) / (o_m_cos_ijk[i,j,k]**2)
                    out_1[i,j,k] = partial_tango_incoming_derivatives_d_de_single(e[j], o_m_cos_ijk[i,j,k])
                    out_2[i,j,k] = partial_tango_incoming_derivatives_d_d_cos_single(e[j], o_m_cos_ijk[i,j,k])
    # out_1 = np.where(
    #     np.logical_or(o_m_cos_ijk <= 0.0, e[np.newaxis,:,np.newaxis] <= 0),
    #     0.0,
    #     0.5 + 0.5 * (1 / np.sqrt(e[np.newaxis, :, np.newaxis]**2 / 4 + e[np.newaxis, :, np.newaxis] * MEC2 / o_m_cos_ijk)) * (e[np.newaxis, :, np.newaxis] / 2 + MEC2 / o_m_cos_ijk[i,j,k])
    # )
    # out_1 = np.where(
    #     np.logical_or(o_m_cos_ijk <= 0.0, e[np.newaxis,:,np.newaxis] <= 0),
    #     0.0,
    #     0.5 * (1 / np.sqrt(e[np.newaxis, :, np.newaxis]**2 / 4 + e[np.newaxis, :, np.newaxis] * MEC2 / o_m_cos_ijk)) * (e[np.newaxis, :, np.newaxis] * MEC2) / (o_m_cos_ijk**2)
    # )
    return (out_1, out_2)


def tango_incoming_sigma(
    e: np.ndarray[float],
    o_m_cos_ijk: np.ndarray[float],
    err_cosines: np.ndarray[float],
    eres: float = 1e-3,
) -> np.ndarray[float]:
    """
    Error of estimated incoming energy using local information
    """
    d_de, d_d_cos = partial_tango_incoming_derivatives(e=e, o_m_cos_ijk=o_m_cos_ijk)
    return np.sqrt((eres * d_de) ** 2 + (err_cosines * d_d_cos) ** 2)


# %% Compton Scattering Formula: Local outbound energy (TANGO)


def tango_outgoing_estimate(
    e: np.ndarray[float], one_minus_cosines: np.ndarray[float]
) -> np.ndarray[float]:
    """Outgoing estimate of energy using TANGO"""
    return tango_incoming_estimate(e, one_minus_cosines) - e


def partial_tango_outgoing_derivatives(
    e: np.ndarray[float], o_m_cos_ijk: np.ndarray[float]
) -> np.ndarray[float]:
    """Partial derivatives of outgoing TANGO energies"""
    d_de, d_d_cos = partial_tango_incoming_derivatives(e, o_m_cos_ijk)
    d_de -= 1.0
    return d_de, d_d_cos


def tango_outgoing_sigma(
    e: np.ndarray[float],
    o_m_cos_ijk: np.ndarray[float],
    err_cosines: np.ndarray[float],
    eres: float = 1e-3,
) -> np.ndarray[float]:
    """Error of estimated outgoing energy using local information"""
    d_de, d_d_cos = partial_tango_outgoing_derivatives(e=e, o_m_cos_ijk=o_m_cos_ijk)
    return np.sqrt((eres * d_de) ** 2 + (err_cosines * d_d_cos) ** 2)


# %% Tabulated cross section data
# Data from NIST XCOM data file for Germanium 32

# fmt: off
# Energies [eV]
sig_energies = np.array([
1.00000E+03, 1.21660E+03, 1.21670E+03, 1.24770E+03, 1.24780E+03, 1.41420E+03,
1.41430E+03, 1.50000E+03, 2.00000E+03, 3.00000E+03, 4.00000E+03, 5.00000E+03,
6.00000E+03, 8.00000E+03, 1.00000E+04, 1.11030E+04, 1.11031E+04, 1.50000E+04,
2.00000E+04, 3.00000E+04, 4.00000E+04, 5.00000E+04, 6.00000E+04, 8.00000E+04,
1.00000E+05, 1.50000E+05, 2.00000E+05, 3.00000E+05, 4.00000E+05, 5.00000E+05,
6.00000E+05, 8.00000E+05, 1.00000E+06, 1.02200E+06, 1.25000E+06, 1.50000E+06,
2.00000E+06, 2.04400E+06, 3.00000E+06, 4.00000E+06, 5.00000E+06, 6.00000E+06,
7.00000E+06, 8.00000E+06, 9.00000E+06, 1.00000E+07, 1.10000E+07, 1.20000E+07,
1.30000E+07, 1.40000E+07, 1.50000E+07, 1.60000E+07, 1.80000E+07, 2.00000E+07,
2.20000E+07, 2.40000E+07, 2.60000E+07, 2.80000E+07, 3.00000E+07, 4.00000E+07,
5.00000E+07, 6.00000E+07, 8.00000E+07, 1.00000E+08, 1.50000E+08, 2.00000E+08,
3.00000E+08, 4.00000E+08, 5.00000E+08, 6.00000E+08, 8.00000E+08, 1.00000E+09,
1.50000E+09, 2.00000E+09, 3.00000E+09, 4.00000E+09, 5.00000E+09, 6.00000E+09,
8.00000E+09, 1.00000E+10, 1.50000E+10, 2.00000E+10, 3.00000E+10, 4.00000E+10,
5.00000E+10, 6.00000E+10, 8.00000E+10, 1.00000E+11
])

# Cross-section [barn/atom]
sig_coherent = np.array([
6.437E+02, 6.281E+02, 6.281E+02, 6.258E+02, 6.258E+02, 6.139E+02, 6.139E+02, 6.072E+02,
5.682E+02, 4.948E+02, 4.313E+02, 3.765E+02, 3.288E+02, 2.522E+02, 1.972E+02, 1.743E+02,
1.743E+02, 1.191E+02, 8.159E+01, 4.586E+01, 2.907E+01, 2.006E+01, 1.475E+01, 9.004E+00,
6.083E+00, 2.903E+00, 1.688E+00, 7.756E-01, 4.431E-01, 2.860E-01, 1.996E-01, 1.129E-01,
7.242E-02, 6.935E-02, 4.643E-02, 3.227E-02, 1.817E-02, 1.740E-02, 8.080E-03, 4.546E-03,
2.910E-03, 2.021E-03, 1.485E-03, 1.137E-03, 8.982E-04, 7.276E-04, 6.013E-04, 5.053E-04,
4.305E-04, 3.712E-04, 3.234E-04, 2.842E-04, 2.246E-04, 1.819E-04, 1.503E-04, 1.263E-04,
1.076E-04, 9.281E-05, 8.085E-05, 4.548E-05, 2.910E-05, 2.021E-05, 1.137E-05, 7.276E-06,
3.234E-06, 1.819E-06, 8.084E-07, 4.547E-07, 2.910E-07, 2.021E-07, 1.137E-07, 7.276E-08,
3.234E-08, 1.819E-08, 8.084E-09, 4.547E-09, 2.910E-09, 2.021E-09, 1.137E-09, 7.276E-10,
3.234E-10, 1.819E-10, 8.084E-11, 4.547E-11, 2.910E-11, 2.021E-11, 1.137E-11, 7.276E-12
])

# Cross-section [barn/atom]
sig_incoherent = np.array([
7.458E-01, 1.016E+00, 1.016E+00, 1.056E+00, 1.056E+00, 1.271E+00, 1.271E+00, 1.381E+00,
2.008E+00, 3.138E+00, 4.126E+00, 5.025E+00, 5.856E+00, 7.328E+00, 8.579E+00, 9.180E+00,
9.180E+00, 1.091E+01, 1.238E+01, 1.400E+01, 1.475E+01, 1.505E+01, 1.512E+01, 1.494E+01,
1.459E+01, 1.357E+01, 1.263E+01, 1.113E+01, 1.003E+01, 9.188E+00, 8.516E+00, 7.497E+00,
6.747E+00, 6.676E+00, 6.038E+00, 5.490E+00, 4.687E+00, 4.629E+00, 3.689E+00, 3.077E+00,
2.658E+00, 2.349E+00, 2.112E+00, 1.922E+00, 1.767E+00, 1.637E+00, 1.527E+00, 1.432E+00,
1.349E+00, 1.276E+00, 1.212E+00, 1.154E+00, 1.054E+00, 9.724E-01, 9.032E-01, 8.440E-01,
7.925E-01, 7.476E-01, 7.079E-01, 5.626E-01, 4.696E-01, 4.046E-01, 3.191E-01, 2.648E-01,
1.886E-01, 1.479E-01, 1.048E-01, 8.209E-02, 6.795E-02, 5.818E-02, 4.544E-02, 3.738E-02,
2.611E-02, 2.021E-02, 1.405E-02, 1.084E-02, 8.867E-03, 7.519E-03, 5.793E-03, 4.730E-03,
3.269E-03, 2.513E-03, 1.733E-03, 1.331E-03, 1.083E-03, 9.159E-04, 7.023E-04, 5.714E-04
])

# Cross-section [barn/atom]
sig_absorption = np.array([
2.275E+05, 1.429E+05, 5.252E+05, 5.988E+05, 8.018E+05, 6.690E+05, 7.574E+05, 6.595E+05,
3.263E+05, 1.154E+05, 5.379E+04, 2.943E+04, 1.786E+04, 8.048E+03, 4.306E+03, 3.206E+03,
2.371E+04, 1.090E+04, 4.996E+03, 1.610E+03, 7.045E+02, 3.670E+02, 2.140E+02, 9.061E+01,
4.624E+01, 1.356E+01, 5.706E+00, 1.733E+00, 7.731E-01, 4.270E-01, 2.699E-01, 1.379E-01,
8.585E-02, 8.158E-02, 5.512E-02, 3.968E-02, 2.437E-02, 2.353E-02, 1.314E-02, 8.813E-03,
6.576E-03, 5.225E-03, 4.325E-03, 3.685E-03, 3.208E-03, 2.839E-03, 2.545E-03, 2.306E-03,
2.107E-03, 1.940E-03, 1.797E-03, 1.674E-03, 1.471E-03, 1.312E-03, 1.184E-03, 1.079E-03,
9.909E-04, 9.160E-04, 8.516E-04, 6.300E-04, 4.998E-04, 4.142E-04, 3.085E-04, 2.457E-04,
1.629E-04, 1.218E-04, 8.098E-05, 6.065E-05, 4.848E-05, 4.037E-05, 3.026E-05, 2.420E-05,
1.612E-05, 1.209E-05, 8.056E-06, 6.041E-06, 4.833E-06, 4.027E-06, 3.020E-06, 2.416E-06,
1.611E-06, 1.208E-06, 8.052E-07, 6.039E-07, 4.831E-07, 4.026E-07, 3.019E-07, 2.416E-07
])

# Cross-section [barn/atom]
sig_pair_nuc = np.array([
0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
0.000E+00, 0.000E+00, 1.073E-02, 5.320E-02, 1.971E-01, 2.116E-01, 5.378E-01, 8.526E-01,
1.130E+00, 1.374E+00, 1.592E+00, 1.788E+00, 1.966E+00, 2.127E+00, 2.274E+00, 2.408E+00,
2.533E+00, 2.648E+00, 2.755E+00, 2.856E+00, 3.040E+00, 3.205E+00, 3.354E+00, 3.490E+00,
3.614E+00, 3.728E+00, 3.833E+00, 4.262E+00, 4.581E+00, 4.831E+00, 5.201E+00, 5.464E+00,
5.883E+00, 6.134E+00, 6.425E+00, 6.592E+00, 6.703E+00, 6.781E+00, 6.886E+00, 6.954E+00,
7.052E+00, 7.106E+00, 7.164E+00, 7.195E+00, 7.215E+00, 7.228E+00, 7.246E+00, 7.257E+00,
7.273E+00, 7.281E+00, 7.290E+00, 7.295E+00, 7.297E+00, 7.299E+00, 7.302E+00, 7.303E+00
])

# Cross-section [barn/atom]
sig_pair_electron = np.array([
0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,
0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 1.291E-03, 5.269E-03,
1.049E-02, 1.611E-02, 2.170E-02, 2.713E-02, 3.232E-02, 3.726E-02, 4.194E-02, 4.636E-02,
5.052E-02, 5.449E-02, 5.825E-02, 6.182E-02, 6.846E-02, 7.450E-02, 8.004E-02, 8.515E-02,
8.987E-02, 9.425E-02, 9.834E-02, 1.153E-01, 1.283E-01, 1.386E-01, 1.542E-01, 1.656E-01,
1.846E-01, 1.966E-01, 2.113E-01, 2.201E-01, 2.262E-01, 2.307E-01, 2.368E-01, 2.409E-01,
2.470E-01, 2.505E-01, 2.543E-01, 2.565E-01, 2.579E-01, 2.589E-01, 2.601E-01, 2.609E-01,
2.620E-01, 2.627E-01, 2.633E-01, 2.637E-01, 2.639E-01, 2.641E-01, 2.642E-01, 2.643E-01
])
# fmt: on

# %% Cross sections
# Interpolation code adapted version from nist-calculators by Mikhail Zelenyi,
# an adaptation of XCOM by NIST


@numba.njit
def fit_absorption(energy_MeV: np.ndarray):
    """
    A least squares fit of a softplus and linear function to the linearized
    data. Roughly 2x speedup over interpolation

    Args:
        - energy_MeV: energy in MeV

    Returns:
        - fit absorption cross section
    """

    # params = np.array([1.323055064485636, -1.482065598945699, 14.074922175534187, -1.011696561692707, 10.35705559222397])
    def softplus(beta, x):
        return 1 / beta * np.log(1 + np.exp(beta * x))

    def approx(
        log_energy_eV,
    ):
        params = np.array(
            [
                1.323055064485636,
                -1.482065598945699,
                14.074922175534187,
                -1.011696561692707,
                10.35705559222397,
            ]
        )
        return params[0] * softplus(1, params[1] * (log_energy_eV - params[2])) + (
            params[3] * log_energy_eV + params[4]
        )

    return np.exp(approx(np.log(energy_MeV * 1e6))) / BARNS_PER_SQCM


@numba.njit
def fit_compton(energy_MeV: np.ndarray):
    """
    A least squares fit of a exponential and linear function to the linearized
    data. Roughly 2x speedup over interpolation

    Args:
        - energy_MeV: energy in MeV

    Returns:
        - fit absorption cross section
    """

    # params = np.array([-5.774955832030122, -0.2543840306526608, 9.371990906611588,-0.9697234863980574, 17.151770815858818])
    def approx(
        log_energy_eV,
    ):
        params = np.array(
            [
                -5.774955832030122,
                -0.2543840306526608,
                9.371990906611588,
                -0.9697234863980574,
                17.151770815858818,
            ]
        )
        return params[0] * np.exp(params[1] * (log_energy_eV - params[2])) + (
            params[3] * log_energy_eV + params[4]
        )

    return np.exp(approx(np.log(energy_MeV * 1e6))) / BARNS_PER_SQCM


@numba.njit
def fit_pair(energy_MeV: np.ndarray | float):
    """
    A least squares fit of a quadratic function to the linearized data. Roughly
    2x speedup over interpolation

    Args:
        - energy_MeV: energy in MeV

    Returns:
        - fit absorption cross section
    """

    # params = np.array([-0.020682536090183546, -5.060366786984834, -8.632083599358173])
    def approx(
        log_energy_eV,
    ):
        params = np.array(
            [-0.020682536090183546, -5.060366786984834, -8.632083599358173]
        )
        return params[0] * log_energy_eV**2 + params[1] * log_energy_eV + params[2]

    energy_eV = energy_MeV * 1e6
    if isinstance(energy_eV, float):
        if energy_eV <= THRESHOLD_PAIR_ATOM:
            return 0.0
        return (
            np.exp(approx(np.log(energy_eV)))
            * ((energy_eV * (energy_eV - THRESHOLD_PAIR_ATOM)) ** 3)
            / BARNS_PER_SQCM
        )
    y = np.zeros((energy_eV.shape))
    idx = energy_eV > THRESHOLD_PAIR_ATOM
    y[idx] = (
        np.exp(approx(np.log(energy_eV[idx])))
        * ((energy_eV[idx] * (energy_eV[idx] - THRESHOLD_PAIR_ATOM)) ** 3)
        / BARNS_PER_SQCM
    )
    return y


def interpolateAbsorptionEdge(
    x: np.ndarray[float],
    y: np.ndarray[float],
    edge: float,
    warn: bool = False,
    linear: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create an interpolator function for photo-absorption including edge
    information.

    If the absorption data has multiple edges, only the largest edge is needed
    here. At that largest edge, the interpolator will switch over to a linear
    interpolation of the data and other edges will be handled using values of x
    and y

    Args:
        - x: x values to interpolate between
        - y: y values to interpolate between
        - edge: largest photo-absorption edge energy for the material
        - warn: print warnings
        - linear: if True, only use linear interpolation

    Adapted version from nist-calculators by Mikhail Zelenyi, an adaptation of
    XCOM by NIST
    """
    linear_interp = interp1d(
        np.log(x), np.log(y), kind="linear", fill_value="extrapolate"
    )

    if linear:

        def linear_interpolator(x_sample: np.ndarray) -> np.ndarray:
            """
            Linear log-log interpolator

            Args:
                - x_sample: sample points for interpolation

            Returns:
                - photo-absorption cross-section
            """
            if (
                np.min(x_sample) < np.min(sig_energies)
                or np.max(x_sample) > np.max(sig_energies)
            ) and warn:
                warnings.warn(
                    "Energy requested is outside of tabulated data, "
                    + "using linear extrapolation",
                    UserWarning,
                )
            return np.exp(linear_interp(np.log(x_sample)))

        return linear_interpolator

    idx = x > edge
    cubic_spline_interp = PchipInterpolator(np.log(x[idx]), np.log(y[idx]))

    def interpolator(x_sample: np.ndarray) -> np.ndarray:
        """
        Cubic spline log-log interpolator

        Args:
            - x_sample: sample points for interpolation

        Returns:
            - photo-absorption cross-section
        """
        if (
            np.min(x_sample) < np.min(sig_energies)
            or np.max(x_sample) > np.max(sig_energies)
        ) and warn:
            warnings.warn(
                "Energy requested is outside of tabulated data, "
                + "using linear extrapolation",
                UserWarning,
            )
        return np.where(
            np.logical_and(x_sample > edge, x_sample < np.max(x)),
            np.exp(cubic_spline_interp(np.log(x_sample))),
            np.exp(linear_interp(np.log(x_sample))),
        )

    return interpolator


def make_log_log_spline(
    x: np.ndarray, y: np.ndarray, warn: bool = False, linear: bool = True
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create spline of log-log data
    """
    log_x = np.log(x)
    log_y = np.log(y)

    linear_interp = interp1d(x=log_x, y=log_y, kind="linear", fill_value="extrapolate")

    if linear:

        def linear_interpolator(x_sample: np.ndarray) -> np.ndarray:
            """
            Linear log-log interpolator

            Args:
                - x_sample: sample points for interpolation

            Returns:
                - cross-section
            """
            if (
                np.min(x_sample) < np.min(sig_energies)
                or np.max(x_sample) > np.max(sig_energies)
            ) and warn:
                warnings.warn(
                    "Energy requested is outside of tabulated data, "
                    + "using linear extrapolation",
                    UserWarning,
                )
            return np.exp(linear_interp(np.log(x_sample)))

        return linear_interpolator

    cubic_spline_interp = PchipInterpolator(x=log_x, y=log_y)

    def interpolator(x_sample: np.ndarray) -> np.ndarray:
        """
        Cubic spline log-log interpolator

        Args:
            - x_sample: sample points for interpolation

        Returns:
            - cross-section
        """
        if (
            np.min(x_sample) < np.min(sig_energies)
            or np.max(x_sample) > np.max(sig_energies)
        ) and warn:
            warnings.warn(
                "Energy requested is outside of tabulated data, "
                + "using linear extrapolation",
                UserWarning,
            )
        return np.where(
            np.logical_and(x_sample > np.min(x), x_sample < np.max(x)),
            np.exp(
                cubic_spline_interp(np.log(x_sample))
            ),  # Use cubic within data range
            np.exp(linear_interp(np.log(x_sample))),  # Extrapolate with linear
        )

    return interpolator


def make_pair_interpolator(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    warn: bool = False,
    linear: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create spline of linearized log-log data
    """

    idx = x > threshold

    linear_interp = interp1d(
        x=np.log(x[idx]),
        y=np.log(y[idx] / (x[idx] * (x[idx] - threshold)) ** 3),
        kind="linear",
        fill_value="extrapolate",
    )

    if linear:

        def linear_interpolator(x_sample: np.ndarray) -> np.ndarray:
            """
            Linearized pair production log-log interpolator

            Args:
                - x_sample: sample points for interpolation

            Returns:
                - pair production cross-section
            """
            if (np.max(x_sample) > np.max(sig_energies)) and warn:
                warnings.warn(
                    "Energy requested is outside of tabulated data, "
                    + "using linear extrapolation",
                    UserWarning,
                )
            idx = x_sample > threshold
            y = np.zeros(x_sample.shape[0])
            y[idx] = (
                np.exp(linear_interp(np.log(x_sample[idx])))
                * (x_sample[idx] * (x_sample[idx] - threshold)) ** 3
            )
            return y

        return linear_interpolator

    cubic_spline_interp = PchipInterpolator(
        x=np.log(x[idx]),
        y=np.log(y[idx] / (x[idx] * (x[idx] - threshold)) ** 3),
    )

    def interpolator(x_sample: np.ndarray) -> np.ndarray:
        """
        Cubic spline linearized pair production log-log interpolator

        Args:
            - x_sample: sample points for interpolation

        Returns:
            - pair production cross-section
        """
        if (np.max(x_sample) > np.max(sig_energies)) and warn:
            warnings.warn(
                "Energy requested is outside of tabulated data, "
                + "using linear extrapolation",
                UserWarning,
            )
        idx = x_sample > threshold
        y = np.zeros(x_sample.shape[0])
        y[idx] = np.where(
            x_sample[idx] < np.max(x),
            np.exp(cubic_spline_interp(np.log(x_sample[idx])))
            * (x_sample[idx] * (x_sample[idx] - threshold)) ** 3,
            np.exp(linear_interp(np.log(x_sample[idx])))
            * (x_sample[idx] * (x_sample[idx] - threshold)) ** 3,
        )
        return y

    return interpolator


__sig_abs = interpolateAbsorptionEdge(
    sig_energies / 1e6, sig_absorption / BARNS_PER_SQCM, K_GE / 1e6
)
__sig_ray = make_log_log_spline(sig_energies / 1e6, sig_coherent / BARNS_PER_SQCM)
__sig_compt = make_log_log_spline(sig_energies / 1e6, sig_incoherent / BARNS_PER_SQCM)
__sig_pair = make_pair_interpolator(
    sig_energies / 1e6, sig_pair_nuc / BARNS_PER_SQCM, THRESHOLD_PAIR_ATOM / 1e6
)


def sig_abs(energies: np.ndarray[float], use_fit: bool = True) -> np.ndarray[float]:
    """Interpolated absorption cross-sections for Germanium"""
    if use_fit:
        return fit_absorption(energies)
    return __sig_abs(energies)


def lin_att_abs(energies: np.ndarray[float], use_fit: bool = True) -> np.ndarray[float]:
    """Interpolated absorption linear attenuation [1/cm]"""
    return sig_abs(energies, use_fit) * RANGE_PROCESS


def sig_ray(energies: np.ndarray[float]) -> np.ndarray[float]:
    """Interpolated rayleigh scattering cross-sections"""
    return __sig_ray(energies)


def lin_att_ray(energies: np.ndarray[float]) -> np.ndarray[float]:
    """Interpolated rayleigh scattering linear attenuation [1/cm]"""
    return sig_ray(energies) * RANGE_PROCESS


def sig_compt(energies: np.ndarray[float], use_fit: bool = True) -> np.ndarray[float]:
    """Interpolated Compton scattering cross-sections"""
    if use_fit:
        return fit_compton(energies)
    return __sig_compt(energies)


def lin_att_compt(
    energies: np.ndarray[float], use_fit: bool = True
) -> np.ndarray[float]:
    """Interpolated Compton scattering linear attenuation [1/cm]"""
    return sig_compt(energies, use_fit) * RANGE_PROCESS


def sig_pair(energies: np.ndarray[float], use_fit: bool = True) -> np.ndarray[float]:
    """Interpolated pair production cross-sections"""
    if use_fit:
        return fit_pair(energies)
    return __sig_pair(energies)


def lin_att_pair(
    energies: np.ndarray[float], use_fit: bool = True
) -> np.ndarray[float]:
    """Interpolated pair production linear attenuation [1/cm]"""
    return sig_pair(energies, use_fit) * RANGE_PROCESS


def sig_total(energies: np.ndarray[float], use_fit: bool = True) -> np.ndarray[float]:
    """Interpolated total cross-section"""
    return (
        sig_abs(energies, use_fit)
        + sig_compt(energies, use_fit)
        + sig_pair(energies, use_fit)
    )


def lin_att_total(
    energies: np.ndarray[float], use_fit: bool = True
) -> np.ndarray[float]:
    """Interpolated total linear attenuation [1/cm]"""
    return (
        sig_abs(energies, use_fit)
        + sig_compt(energies, use_fit)
        + sig_pair(energies, use_fit)
    ) * RANGE_PROCESS

@numba.njit
def lin_att_total_fit(energies: np.ndarray[float]) -> np.ndarray[float]:
    """Interpolated total linear attenuation [1/cm]"""
    return (
        fit_absorption(energies)
        + fit_compton(energies)
        + fit_pair(energies)
    ) * RANGE_PROCESS


def range_process(sigma: np.ndarray[float]) -> np.ndarray[float]:
    """
    Given gamma macroscopic cross section (absorption, Compton scattering, or
    pair production), return linear attenuation coefficient.
    """
    # 1/lambda = N_AV [atoms/mol] * RHO_GE [g/cm^3] / A_GE [g/mol] * sigma [cm2/atom] = [1/cm]
    return (sigma * N_AV * RHO_GE) / A_GE


def proba(
    lamb_inv: np.ndarray[float], distance: np.ndarray[float]
) -> np.ndarray[float]:
    """
    Cumulative exponential distribution: interaction probability. What is the
    probability that the interaction would occur at a distance greater than the
    provided distance? If value is small, the distance is probably too long. We
    can evaluate where, e.g., 90% of interactions should occur by looking for
    proba=0.1
    lamb_inv is a mean free distance.
    """
    return np.exp(-distance * lamb_inv)


# %% Klein-Nishina formula
@numba.njit
def KN_differential_cross_single(
    E_imo: float,
    one_minus_cos_theta: float,
    Ei: float = None,
    sigma_compt: float = None,
    relative: bool = True,
    integrate: bool = False,
) -> float:
    """
    Vectorized (relative) Klein-Nishina differential cross-section

    Args:
        - E_imo: incoming gamma-ray energy
        - one_minus_cos_theta: 1 - cos(theta) of the scattering angle theta
        - sigma_compt: Compton scattering cross-section at energy E_imo
        - Ei: outgoing energy
        - relative: divide by the total Compton scattering cross-section value

    Returns:
        - Klein-Nishina differential cross-section value
    """
    if sigma_compt is None and relative:
        sigma_compt = fit_compton(E_imo)
    if E_imo <= 0:
        return 0.0
    if Ei is not None:
        ll = Ei / E_imo
    else:
        ll = 1 / (1 + E_imo / MEC2 * (one_minus_cos_theta))
    sin_sq = 1 - (1 - one_minus_cos_theta) ** 2
    out = 0.5 * (R_0**2) * (ll**2) * (ll + 1 / ll - sin_sq)
    if integrate:
        out *= 2 * np.pi * np.sqrt(sin_sq)  # Integrate with respect to phi
    if relative:
        out /= sigma_compt
    return out

@numba.njit
def KN_differential_cross(
    E_imo: np.ndarray[float],
    one_minus_cos_theta: np.ndarray[float],
    Ei: np.ndarray[float] = None,
    sigma_compt: np.ndarray[float] = None,
    relative: bool = True,
    integrate: bool = False,
) -> np.ndarray[float]:
    """
    Klein-Nishina differential cross-section

    Args:
        - E_imo: incoming gamma-ray energy
        - one_minus_cos_theta: 1 - cos(theta) of the scattering angle theta
        - sigma_compt: Compton scattering cross-section at energy E_imo
        - Ei: outgoing energy
        - relative: divide by the total Compton scattering cross-section value

    Returns:
        - Klein-Nishina differential cross-section value
    """
    out = np.zeros((len(E_imo),))
    for i in numba.prange(len(E_imo)):
        if Ei is not None:
            if sigma_compt is not None:
                out[i] = KN_differential_cross_single(E_imo[i], one_minus_cos_theta[i], Ei[i], sigma_compt[i], relative, integrate)
            else:
                out[i] = KN_differential_cross_single(E_imo[i], one_minus_cos_theta[i], Ei[i], None, relative, integrate)
        else:
            if sigma_compt is not None:
                out[i] = KN_differential_cross_single(E_imo[i], one_minus_cos_theta[i], None, sigma_compt[i], relative, integrate)
            else:
                out[i] = KN_differential_cross_single(E_imo[i], one_minus_cos_theta[i], None, None, relative, integrate)
    return out


# %% Old cross-sections
# def make_sig_ray_l_interp():
#     """
#     Create the linear interpolator
#     """
#     arr = np.loadtxt('data/physics/RayleighCrossSection-z32.csv',
#                  delimiter=",", dtype=str)
#     xp = arr[1:,0].astype(np.float64)*10**(-6)
#     fp = arr[1:,1].astype(np.float64)*10**(-24)
#     return interp1d(np.log(xp), np.log(fp), kind='slinear', fill_value='extrapolate')

# sig_ray_l_interp = make_sig_ray_l_interp()

# def sig_ray(E):
#     """
#     Use linear interpolator to do logarithmic interpolation
#     """
#     return np.exp(sig_ray_l_interp(np.log(E)))

# def make_sig_ray_livermore_l_interp():
#     """
#     Make linear interpolator
#     """
#     arr = np.loadtxt('data/physics/rayleigh_geant4_re-cs-32.csv',
#                  delimiter=",", dtype=str)
#     xp = arr[1:,0].astype(np.float64)
#     fp = arr[1:,1].astype(np.float64)/100 # convert to cm^2/atom
#     return interp1d(np.log(xp), np.log(fp), kind='slinear', fill_value='extrapolate')

# sig_ray_livermore_l_interp = make_sig_ray_livermore_l_interp()

# def sig_ray_livermore(E):
#     """
#     Use logarithmic interpolation
#     """
#     return np.exp(sig_ray_livermore_l_interp(np.log(E)))/E**2

# # def sig_ray_livermore(E):
# #     """
# #     Cross section of Rayleigh scattering using the Livermore model with
# #     coefficients from GEANT4
# #     """
# #     dataframe = pd.read_csv('data/physics/rayleigh_geant4_re-cs-32.csv')
# #     xp = np.array(dataframe['Photon energy (MeV)'])
# #     fp = np.array(dataframe['Cross section (mm*mm)'])/10/10 # convert to cm^2/atom
# #     out = log_interp(E, xp, fp)/E**2
# #     return out


# @njit
def sig_abs_old(E: np.ndarray[float]) -> np.ndarray[float]:
    """
    Cross section of photoelectric absorption with energy E.

    See [Wikipedia: Gamma ray cross
    section](https://en.wikipedia.org/wiki/Gamma_ray_cross_section) for more
    information.

    Arg:
        E : energy of the photon [MeV]
    Returns:
        sig_abs : the cross section of absorption at E [cm^2/atom]
    """
    hnu_k = (Z_GE - 0.03) ** 2 * MEC2 * ALPHA**2 / 2
    # if isinstance(E, (float, int)):
    #     E = np.array(E)
    # return np.piecewise(E,
    #                     (E >= 0.025,
    #                      np.logical_and(E < 0.025, E >= 0.0111),
    #                      np.logical_and(E < 0.0111, E > 0),
    #                      E <= 0),
    #                     ((4 * ALPHA**4 * np.sqrt(2) * 6.651e-25 * Z_GE**5) / ((E/MEC2)**3),
    #                      np.power((hnu_k/E),2.6666) * 2.2 * 6.3e-18 / Z_GE**2,
    #                      np.power((hnu_k/E),2.6666) * 2.2 * 6.3e-18 / Z_GE**2/8.5,
    #                      0.0))
    sigma_abs = np.zeros(E.shape)
    ind = E >= 0.025
    sigma_abs[ind] = (4 * ALPHA**4 * np.sqrt(2) * 6.651e-25 * Z_GE**5) / (
        (E[ind] / MEC2) ** 3
    )
    ind = np.logical_and(E < 0.025, E > 0)
    sigma_abs[ind] = np.power((hnu_k / E[ind]), 2.6666) * 2.2 * 6.3e-18 / Z_GE**2
    ind = np.logical_and(E < 0.0111, E > 0)
    sigma_abs[ind] = sigma_abs[ind] / 8.5
    return sigma_abs


# def sig_abs_scofield(E:np.ndarray[float]) -> np.ndarray[float]:
#     """
#     Absorption cross section
#     """
#     if isinstance(E, (float, int)):
#         E = np.array(E)
#     dataframe = pd.read_csv('data/physics/absorption_Scofield1973.csv')
#     xp = np.array(dataframe['Photon Energy (keV)'])*10**(-3)
#     fp = np.array(dataframe['Cross-section (barns/atom)'])*10**(-24)
#     out = log_interp(E, xp, fp)
#     return out

# def sig_abs_sandia(E):
#     """Sandia version of absorption cross-section from GEANT4"""
#     sandia = np.array([[ 0.01,         0.4941E+04,  0.0000E+00,  0.0000E+00,  0.0000E+00 ] ,
#                         [ 0.0305,       0.1269E+05, -0.9339E+03,  0.2767E+02, -0.1952E+00 ] ,
#                         [ 0.1,         -0.1358E+04,  0.3695E+04, -0.4569E+03,  0.1599E+02 ] ,
#                         [ 1.217,       -0.7402E+04,  0.3636E+05, -0.6670E+05,  0.5346E+05 ] ,
#                         [ 1.248,       -0.3400E+03,  0.3232E+04,  0.1163E+05, -0.1089E+04 ] ,
#                         [ 1.413,       -0.5940E+02,  0.1646E+04,  0.2621E+05, -0.1507E+05 ] ,
#                         [ 11.104,      -0.2025E+01,  0.1283E+03,  0.4069E+06, -0.1510E+07 ] ,
#                         [ 100.0,        0.6478E+00, -0.2603E+03,  0.4045E+06, -0.5153E+05 ] ,
#                         [ 500.0,        0.2155E+00,  0.2609E+03,  0.2074E+06,  0.2220E+08 ]])
#     energy_ranges = sandia[:,0]
#     cross_section = np.zeros(E.shape)
#     for i, (energy_low, energy_high) in enumerate(zip(energy_ranges,
#                                                       list(energy_ranges[1:]) + [np.inf])):
#         ind = np.logical_and(energy_low <= E*1000, E*1000 < energy_high)
#         cross_section[ind] = sandia[i,1]/(E[ind]*1000) + \
#             sandia[i,2]/(E[ind]*1000)**2 + \
#             sandia[i,3]/(E[ind]*1000)**3 + \
#             sandia[i,4]/(E[ind]*1000)**4
#     return RHO_GE*cross_section/range_process(1)


# @njit
def sig_compt_old(E: np.ndarray[float]) -> np.ndarray[float]:
    """
    Cross section of Compton scattering with energy E.

    See [Wikipedia: Gamma ray cross
    section](https://en.wikipedia.org/wiki/Gamma_ray_cross_section#Compton_scattering_cross_section)
    for more information.

    This function comes from the integration of the Klein-Nishina formula
    (integrates with respect to azimuthal and polar angle).

    Arg:
        E : energy of the photon [MeV]
    Returns:
        sig_compt : the cross section of Compton scattering at E [cm^2/atom]
    """
    # if isinstance(E, (float, int)):
    #     E = np.array(E)
    ind = E > 0
    sigma_compt = np.zeros(E.shape)
    gamma = E[ind] / MEC2
    sigma_compt[ind] = (2 * np.pi * (R_0) ** 2 * Z_GE) * (
        ((1 + gamma) / gamma**2)
        * ((2 * (1 + gamma) / (1 + 2 * gamma)) - (np.log(1 + 2 * gamma) / gamma))
        + (
            np.log(1 + 2 * gamma) / (2 * gamma)
            - ((1 + 3 * gamma) / ((1 + 2 * gamma) ** 2))
        )
    )
    return sigma_compt


# @njit
def sig_pair_old(E: np.ndarray[float]) -> np.ndarray[float]:
    """
    Cross section of pair production with energy E.

    See [Wikipedia: Gamma ray cross
    section](https://en.wikipedia.org/wiki/Gamma_ray_cross_section) for more
    information.

    Arg:
        E: energy of the photon [MeV]
    Returns:
        sig_pair: the cross section of pair production at E [cm^2/atom]
    """
    # if isinstance(E, (float, int)):
    #     E = np.array(E)
    sigma_pair = np.zeros(E.shape)
    ind = np.logical_and(E >= 1.022, E < 1.15)
    sigma_pair[ind] = (1 - ((1.15 - E[ind]) / 0.129)) * 7.55e-28 * 1e-24
    ind = E > 1.15
    sigma_pair[ind] = (
        0.792189
        * np.log(E[ind] + 0.948261 - 1.1332 * E[ind] + 0.15567 * E[ind] ** 2)
        * 1e-24
    )
    return sigma_pair
