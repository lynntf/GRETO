"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Permutation level values
"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional # Tuple

import numba
import numpy as np

import greto.physics as phys
from greto.fast_features.event_level import event_level_values
from greto.fom_tools import escape_probability as ft_escape_probability
from greto.utils import cumsum, perm_to_transition, reverse_cumsum

perm_level_values = namedtuple(
    "perm_level_values",
    [
        # "transition_1D",
        "transition_2D",
        "transition_3D",
        "energies_perm",
        "energy_sum",
        # "energy_cumsum",
        "energy_cumsum_with_zero",
        "energy_rev_cumsum",
        "Ns",
        # "energy_rev_cumsum_sigma",
        "distance_perm",
        "ge_distance_perm",
        "cos_act_perm",
        "cos_err_perm",
        "theta_act_perm",
        "theta_err_perm",
        "local_tango_estimates_perm",
        "tango_estimates_perm",
        "tango_estimates_sigma_perm",
        # "estimate_start_energy_perm",
        # "sqrt_N",
        # "estimate_start_energy_sigma_weighted_perm",
        "cos_theor_perm",
        "cos_theor_sigma_perm",
        "theta_theor_perm",
        "theta_theor_sigma_perm",
        "scattered_energy",
        # "scattered_energy_sigma",
        "res_sum_geo",
        "res_sum_geo_sigma",
        "res_sum_geo_v",
        "res_sum_loc",
        "d_de",
        "d_d_cos",
        "res_sum_loc_sigma",
        "res_sum_loc_v",
        "res_loc_geo",
        "res_loc_geo_sigma",
        "res_loc_geo_v",
        "res_cos",
        "res_cos_cap",
        "res_cos_sigma",
        "res_cos_v",
        "res_cos_cap_v",
        "res_theta",
        "res_theta_cap",
        "res_theta_sigma",
        "res_theta_v",
        "res_theta_cap_v",
        "compton_penalty",
        "compton_penalty_ell1",
        "linear_attenuation_abs",
        "linear_attenuation_compt",
        "linear_attenuation_pair",
        "lin_mu_total",
        "klein_nishina_use_Ei",
        "klein_nishina",
        # "klein_nishina_differential_cross_section_use_Ei",
        # "klein_nishina_differential_cross_section",
        "klein_nishina_relative_use_Ei",
        "klein_nishina_relative",
        # "klein_nishina_differential_cross_section_relative_use_Ei",
        # "klein_nishina_differential_cross_section_relative",
        "radii_perm",
        "escape_probability",
    ],
)


@numba.njit
def theta_err_perm_func(cos_err, cos_act):
    """Standard error in actual theta"""
    return cos_err / np.sqrt(1.0 - cos_act**2)


@numba.njit
def estimate_start_energy_perm_func(
    energy_sum, final_energy, use_threshold, energy_cumsum_with_zero, tango_estimates
):
    """Estimate the start energy using (unweighted) TANGO"""
    out = np.mean(energy_cumsum_with_zero[:-2] + tango_estimates)
    if use_threshold:
        if (
            out < energy_sum
            or phys.njit_cos_theor(out - energy_sum + final_energy, out - energy_sum)
            < -1
        ):
            out = energy_sum
    return out


@numba.njit
def estimate_start_energy_sigma_weighted_perm_func(
    energy_sum,
    final_energy,
    use_threshold,
    energy_cumsum_with_zero,
    local_tango_estimates_perm,
    tango_estimates_sigma_perm,
    sqrt_N,
    eres,
):
    """Estimate the start energy using TANGO"""
    out = np.sum(
        (energy_cumsum_with_zero[:-2] + local_tango_estimates_perm)
        / (tango_estimates_sigma_perm + eres * sqrt_N)
    ) / np.sum(1.0 / (tango_estimates_sigma_perm + eres * sqrt_N))
    if use_threshold:
        if (
            out < energy_sum
            or phys.njit_cos_theor(out - energy_sum + final_energy, out - energy_sum)
            < -1
        ):
            out = energy_sum
    return out


@numba.njit
def theta_theor_sigma_perm_func(cos_theor_sigma_perm, cos_theor_perm):
    """Standard error of theoretical theta"""
    return cos_theor_sigma_perm / np.sqrt(np.abs(1 - cos_theor_perm**2))


@numba.njit
def res_sum_geo_sigma_func(scattered_energy, energy_rev_cumsum, Ns, cos_err_perm, eres):
    """Residual between energy sums and geometrical energy (CSF)"""
    return np.sqrt(
        eres**2
        * (
            (scattered_energy / energy_rev_cumsum[:-1]) ** 4
            + Ns[:-2] * (1 - (scattered_energy / energy_rev_cumsum[:-1]) ** 2) ** 2
        )
        + (cos_err_perm * (scattered_energy**2 / phys.MEC2)) ** 2
    )


@numba.njit
def res_sum_loc_sigma_func(d_de, Ns, cos_err_perm, d_d_cos, eres):
    """Residual between energy sums and local incoming energy estimate"""
    return np.sqrt(
        (eres**2 * ((1 - d_de) ** 2 + Ns[:-2] - 1)) + (cos_err_perm * (d_d_cos)) ** 2
    )


@numba.njit
def res_loc_geo_func(local_tango_estimates_perm, energies_perm, scattered_energy):
    """
    Residual between local outgoing (incoming - e) energy estimates and
    geometrical energy (CSF)
    """
    return np.abs((local_tango_estimates_perm - energies_perm[:-1]) - scattered_energy)


@numba.njit
def res_loc_geo_sigma_func(
    d_de, scattered_energy, energy_rev_cumsum, Ns, cos_err_perm, d_d_cos, eres
):
    """Standard error for residual between local outgoing energy and geometrical energy (CSF)"""
    return np.sqrt(
        eres**2
        * (
            (d_de - (scattered_energy / energy_rev_cumsum[1:]) ** 2) ** 2
            + Ns[:-2] * (scattered_energy / energy_rev_cumsum[1:]) ** 4
        )
        + cos_err_perm**2 * (d_d_cos - scattered_energy**2 / phys.MEC2) ** 2
    )


@numba.njit
def res_theta_func(fix_nan, theta_act_perm, theta_theor_perm):
    """Residual between geometrical scattering angle and theoretical"""
    out = np.abs(theta_act_perm - theta_theor_perm)
    if fix_nan is not None:
        out[np.isnan(out)] = fix_nan
    return out


@numba.njit
def res_theta_cap_func(fix_nan, theta_act_perm, cos_theor_perm):
    """Residual between geometrical scattering angle and theoretical capped at cos theta = -1"""
    out = np.abs(theta_act_perm - np.arccos(np.maximum(cos_theor_perm, -1)))
    if fix_nan is not None:
        out[np.isnan(out)] = fix_nan
    return out


@numba.njit
def res_theta_sigma_func(theta_err_perm, theta_theor_sigma_perm, eres):
    """Standard error of angle residual"""
    return np.sqrt(theta_err_perm**2 + (eres * theta_theor_sigma_perm) ** 2)


@numba.njit
def res_cos_sigma_func(cos_err_perm, cos_theor_sigma_perm, eres):
    """Standard error of cosine residual"""
    return np.sqrt(cos_err_perm**2 + (eres * cos_theor_sigma_perm) ** 2)


@numba.njit
def res_cos_cap_func(cos_act_perm, cos_theor_perm):
    """Residual between geometrical cosine and theoretical capped at cos theta = -1"""
    return np.abs(cos_act_perm - np.maximum(cos_theor_perm, -1))


def perm_atoms(
    permutation: tuple[int],
    event_calc: event_level_values,
    start_point: int = 0,
    start_energy: float = None,
    use_threshold: bool = False,
    Nmi: Optional[int] = None,
    fix_nan: float = 2 * np.pi,
    outer_radius: float = 32.5,
    number_of_values: int = 64,
    name_mode: bool = False,
    dependency_mode: bool = False,
    tango_mode: bool = False,
    all_computations: bool = False,
    boolean_vector: np.ndarray = None,
    eres: float = 1e-3,
):
    """
    Inputs are general computational atoms

    Outputs are permuted computational atoms
    """
    compute_mode = not name_mode and not dependency_mode
    if compute_mode and boolean_vector is None:
        all_computations = True
    if all_computations:
        boolean_vector = np.ones((number_of_values,), dtype=np.bool_)

    def compute_value(name, dependencies, compute_fn):
        """
        Computes the value of the
        """
        if compute_mode:
            if boolean_vector[index] or all_computations:
                return compute_fn()
            return None
        elif name_mode:
            names.append(name)
        elif dependency_mode:
            dependencies_dict[name] = dependencies

    names = []
    dependencies_dict = {}
    index = 0

    if compute_mode:
        if Nmi is None:
            Nmi = len(permutation)
        if start_point is not None:
            permutation = tuple(
                [start_point] + list(permutation)
            )  # include the start point; 0

        transition_1D = np.array(permutation)
        if start_energy is None:
            # start_energy != energy_sum if only a partial permutation (or TANGO; later)
            start_energy = np.sum(event_calc.energy_matrix[transition_1D[1:]])

    # if compute_mode and not njit_any(boolean_vector[index : index + 7]):
    #     index += 7
    #     energies_perm = None
    #     energy_sum = None
    #     transition_2D = None
    #     transition_3D = None
    #     # energy_cumsum = None
    #     energy_cumsum_with_zero = None
    #     energy_rev_cumsum = None
    #     Ns = None
    #     # energy_rev_cumsum_sigma = None
    # else:
    energies_perm = compute_value(
        "energies_perm",
        ["energy_matrix"],
        lambda: event_calc.energy_matrix[transition_1D[1:]],
    )
    index += 1

    energy_sum = compute_value("energy_sum", [], lambda: np.sum(energies_perm))
    index += 1

    transition_2D = compute_value(
        "transition_2D", [], lambda: perm_to_transition(permutation, D=2)
    )  # transition indices for array lookups)
    index += 1

    transition_3D = compute_value(
        "transition_3D", [], lambda: perm_to_transition(permutation, D=3)
    )  # transition indices for array lookups)
    index += 1

    # energy_cumsum = compute_value(
    #     "energy_cumsum", ["energies_perm"],
    #     lambda: cumsum(energies_perm))
    # index += 1

    energy_cumsum_with_zero = compute_value(
        "energy_cumsum_with_zero",
        ["energy_matrix"],
        lambda: cumsum(event_calc.energy_matrix[transition_1D]),
    )
    index += 1

    energy_rev_cumsum = compute_value(
        "energy_rev_cumsum",
        ["energies_perm", "energy_sum", "energy_matrix"],
        lambda: reverse_cumsum(energies_perm) + start_energy - energy_sum,
    )
    index += 1

    Ns = compute_value(
        "Ns", [], lambda: np.arange(Nmi, Nmi - len(permutation), -1, dtype=int)
    )
    index += 1

    # energy_rev_cumsum_sigma = compute_value(
    #     "energy_rev_cumsum_sigma", ["Ns"],
    #     lambda: eres * np.sqrt(Ns[:-1]))
    # index += 1

    # if compute_mode and not njit_any(boolean_vector[index : index + 2]):
    #     index += 2
    #     distance_perm = None
    #     ge_distance_perm = None
    # else:
    distance_perm = compute_value(
        "distance_perm",
        ["distance", "transition_2D"],
        lambda: event_calc.distance[transition_2D],
    )
    index += 1

    ge_distance_perm = compute_value(
        "ge_distance_perm",
        ["ge_distance", "transition_2D"],
        lambda: event_calc.ge_distance[transition_2D],
    )
    index += 1

    # if compute_mode and not njit_any(boolean_vector[index : index + 4]):
    #     index += 4
    #     cos_act_perm = None
    #     cos_err_perm = None
    #     theta_act_perm = None
    #     theta_err_perm = None
    # else:
    cos_act_perm = compute_value(
        "cos_act_perm",
        ["cos_act", "transition_3D"],
        lambda: event_calc.cos_act[transition_3D],
    )
    index += 1

    cos_err_perm = compute_value(
        "cos_err_perm",
        ["cos_err", "transition_3D"],
        lambda: event_calc.cos_err[transition_3D],
    )
    index += 1

    theta_act_perm = compute_value(
        "theta_act_perm",
        ["cos_act", "transition_3D"],
        lambda: np.arccos(event_calc.cos_act[transition_3D]),
    )
    index += 1

    theta_err_perm = compute_value(
        "theta_err_perm",
        ["cos_err", "transition_3D"],
        # lambda: cos_err[transition_3D] / np.sqrt(1.0 - cos_act[transition_3D]**2))
        lambda: theta_err_perm_func(
            event_calc.cos_err[transition_3D], event_calc.cos_act[transition_3D]
        ),
    )
    index += 1

    # if compute_mode and not njit_any(boolean_vector[index : index + 5]):
    #     index += 5
    #     local_tango_estimates_perm = None
    #     tango_estimates_perm = None
    #     tango_estimates_sigma_perm = None
    #     # estimate_start_energy_perm = None
    #     sqrt_N = None
    #     estimate_start_energy_sigma_weighted_perm = None
    # else:
    local_tango_estimates_perm = compute_value(
        "local_tango_estimates_perm",
        ["tango_estimates", "transition_3D"],
        lambda: event_calc.tango_estimates[transition_3D],
    )
    index += 1

    tango_estimates_perm = compute_value(
        "tango_estimates_perm",
        ["local_tango_estimates_perm", "energy_cumsum_with_zero"],
        lambda: energy_cumsum_with_zero[:-2] + local_tango_estimates_perm,
    )
    index += 1

    tango_estimates_sigma_perm = compute_value(
        "tango_estimates_sigma_perm",
        ["tango_estimates_sigma", "transition_3D"],
        lambda: event_calc.tango_estimates_sigma[transition_3D],
    )
    index += 1

    # estimate_start_energy_perm = compute_value(
    #     "estimate_start_energy_perm", ["energy_sum", "energies_perm", "energy_cumsum_with_zero", "tango_estimates", "transition_3D"],
    #     lambda: estimate_start_energy_perm_func(energy_sum, energies_perm[-1], use_threshold, energy_cumsum_with_zero, event_calc.tango_estimates[transition_3D]))
    # index += 1

    # energy_rev_cumsum_tango = compute_value(
    #     "energy_rev_cumsum_tango", ["energies_perm", "estimate_start_energy_perm", "energy_matrix"],
    #     lambda: reverse_cumsum(energies_perm) + estimate_start_energy_perm - energy_sum)
    # index += 1

    sqrt_N = compute_value(
        "sqrt_N", [], lambda: np.sqrt(np.arange(1, len(permutation) - 1, 1))
    )
    index += 1

    estimate_start_energy_sigma_weighted_perm = compute_value(
        "estimate_start_energy_sigma_weighted_perm",
        [
            "energy_sum",
            "energies_perm",
            "energy_cumsum_with_zero",
            "local_tango_estimates_perm",
            "tango_estimates_sigma_perm",
            "sqrt_N",
        ],
        lambda: estimate_start_energy_sigma_weighted_perm_func(
            energy_sum,
            energies_perm[-1],
            use_threshold,
            energy_cumsum_with_zero,
            local_tango_estimates_perm,
            tango_estimates_sigma_perm,
            sqrt_N,
            eres,
        ),
    )
    index += 1

    if tango_mode:
        start_energy = estimate_start_energy_sigma_weighted_perm
        energy_rev_cumsum = reverse_cumsum(energies_perm) + start_energy - energy_sum

    # energy_rev_cumsum_tango_sigma_weighted = compute_value(
    #     "energy_rev_cumsum_tango_sigma_weighted", ["energies_perm", "estimate_start_energy_sigma_weighted_perm", "energy_matrix"],
    #     lambda: reverse_cumsum(energies_perm) + estimate_start_energy_sigma_weighted_perm - energy_sum)
    # index += 1

    # tango_used = False
    # if compute_mode:
    #     if estimate_start_energy_sigma_weighted_perm is not None and energy_sum is not None:
    #         if estimate_start_energy_sigma_weighted_perm > energy_sum:
    #             tango_used = True

    # if compute_mode and not njit_any(boolean_vector[index : index + 5]):
    #     index += 5
    #     cos_theor_perm = None
    #     cos_theor_sigma_perm = None
    #     theta_theor_perm = None
    #     theta_theor_sigma_perm = None
    #     scattered_energy = None
    # else:
    cos_theor_perm = compute_value(
        "cos_theor_perm",
        ["energy_rev_cumsum"],
        lambda: phys.cos_theor_sequence(energy_rev_cumsum),
    )
    index += 1

    # def cos_theor_perm_tango_func(tango_used, cos_theor_perm, energy_rev_cumsum_tango_sigma_weighted):
    #     if tango_used:
    #         return phys.cos_theor_sequence(energy_rev_cumsum_tango_sigma_weighted)
    #     return cos_theor_perm

    # cos_theor_perm_tango = compute_value(
    #     "cos_theor_perm_tango", ["cos_theor_perm", "energy_rev_cumsum_tango_sigma_weighted"],
    #     lambda: cos_theor_perm_tango_func(tango_used, cos_theor_perm, energy_rev_cumsum_tango_sigma_weighted) )
    # index += 1

    cos_theor_sigma_perm = compute_value(
        "cos_theor_sigma_perm",
        ["energy_rev_cumsum"],
        lambda: phys.cos_theor_sigma(
            energy_rev_cumsum[:-1], energy_rev_cumsum[1:], Nmi - 1
        ),
    )
    index += 1

    # def cos_theor_sigma_perm_tango_func(tango_used, cos_theor_sigma_perm, energy_rev_cumsum_tango_sigma_weighted):
    #     if tango_used:
    #         return phys.cos_theor_sigma(energy_rev_cumsum_tango_sigma_weighted[:-1], energy_rev_cumsum_tango_sigma_weighted[1:], Nmi - 1)
    #     return cos_theor_sigma_perm

    # cos_theor_sigma_perm_tango = compute_value(
    #     "cos_theor_sigma_perm_tango", ["cos_theor_sigma_perm", "energy_rev_cumsum_tango_sigma_weighted"],
    #     lambda: cos_theor_sigma_perm_tango_func(tango_used, cos_theor_sigma_perm, energy_rev_cumsum_tango_sigma_weighted))
    # index += 1

    theta_theor_perm = compute_value(
        "theta_theor_perm",
        ["energy_rev_cumsum"],
        lambda: phys.theta_theor(energy_rev_cumsum[:-1], energy_rev_cumsum[1:]),
    )
    index += 1

    # def theta_theor_perm_tango_func(tango_used, theta_theor_perm, energy_rev_cumsum_tango_sigma_weighted):
    #     if tango_used:
    #         return phys.theta_theor(energy_rev_cumsum_tango_sigma_weighted[:-1], energy_rev_cumsum_tango_sigma_weighted[1:])
    #     return theta_theor_perm

    # theta_theor_perm_tango = compute_value(
    #     "theta_theor_perm_tango", ["theta_theor_perm", "energy_rev_cumsum_tango_sigma_weighted"],
    #     lambda: theta_theor_perm_tango_func(tango_used, theta_theor_perm, energy_rev_cumsum_tango_sigma_weighted))
    # index += 1

    theta_theor_sigma_perm = compute_value(
        "theta_theor_sigma_perm",
        ["cos_theor_sigma_perm", "cos_theor_perm"],
        lambda: theta_theor_sigma_perm_func(cos_theor_sigma_perm, cos_theor_perm),
    )
    index += 1

    scattered_energy = compute_value(
        "scattered_energy",
        ["energy_rev_cumsum", "cos_act_perm"],
        lambda: phys.outgoing_energy_csf(energy_rev_cumsum[:-1], 1 - cos_act_perm),
    )
    index += 1

    # def scattered_energy_tango_func(tango_used, scattered_energy, energy_rev_cumsum_tango_sigma_weighted, cos_act_perm):
    #     if tango_used:
    #         return phys.outgoing_energy_csf(energy_rev_cumsum_tango_sigma_weighted[:-1], 1 - cos_act_perm)
    #     return scattered_energy

    # scattered_energy_tango = compute_value(
    #     "scattered_energy_tango", ["scattered_energy", "energy_rev_cumsum_tango_sigma_weighted", "cos_act_perm"],
    #     lambda: scattered_energy_tango_func(tango_used, scattered_energy, energy_rev_cumsum_tango_sigma_weighted, cos_act_perm))
    # index += 1

    # scattered_energy_sigma = compute_value(
    #     "scattered_energy_sigma", ["energy_rev_cumsum", "cos_act_perm", "cos_err_perm", "energy_rev_cumsum_sigma"],
    #     lambda: phys.outgoing_energy_csf_sigma(energy_rev_cumsum[:-1], 1 - cos_act_perm, cos_err_perm, energy_rev_cumsum_sigma[1:]))
    # index += 1

    # def scattered_energy_sigma_tango_func(tango_used, scattered_energy_sigma, energy_rev_cumsum_tango_sigma_weighted, cos_act_perm, cos_err_perm, energy_rev_cumsum_sigma):
    #     if tango_used:
    #         return phys.outgoing_energy_csf_sigma(energy_rev_cumsum_tango_sigma_weighted[:-1], 1 - cos_act_perm, cos_err_perm, energy_rev_cumsum_sigma[1:])
    #     return scattered_energy_sigma

    # scattered_energy_sigma_tango = compute_value(
    #     "scattered_energy_sigma_tango", ["scattered_energy_sigma", "energy_rev_cumsum_tango_sigma_weighted", "cos_act_perm", "cos_err_perm", "energy_rev_cumsum_sigma"],
    #     lambda: scattered_energy_sigma_tango_func(tango_used, scattered_energy_sigma, energy_rev_cumsum_tango_sigma_weighted, cos_act_perm, cos_err_perm, energy_rev_cumsum_sigma))
    # index += 1

    # if compute_mode and not njit_any(boolean_vector[index : index + 3]):
    #     index += 3
    #     res_sum_geo = None
    #     res_sum_geo_sigma = None
    #     res_sum_geo_v = None
    # else:
    res_sum_geo = compute_value(
        "res_sum_geo",
        ["energy_rev_cumsum", "scattered_energy"],
        lambda: np.abs(energy_rev_cumsum[1:] - scattered_energy),
    )
    index += 1

    res_sum_geo_sigma = compute_value(
        "res_sum_geo_sigma",
        ["scattered_energy", "energy_rev_cumsum", "Ns", "cos_err_perm"],
        lambda: res_sum_geo_sigma_func(
            scattered_energy, energy_rev_cumsum, Ns, cos_err_perm, eres
        ),
    )
    index += 1

    res_sum_geo_v = compute_value(
        "res_sum_geo_v",
        ["res_sum_geo", "res_sum_geo_sigma"],
        lambda: res_sum_geo / res_sum_geo_sigma,
    )
    index += 1

    # if compute_mode and not njit_any(boolean_vector[index : index + 5]):
    #     index += 5
    #     res_sum_loc = None
    #     d_de = None
    #     d_d_cos = None
    #     res_sum_loc_sigma = None
    #     res_sum_loc_v = None
    # else:
    res_sum_loc = compute_value(
        "res_sum_loc",
        ["energy_rev_cumsum", "local_tango_estimates_perm"],
        lambda: np.abs(energy_rev_cumsum[:-1] - local_tango_estimates_perm),
    )
    index += 1

    d_de = compute_value(
        "d_de",
        ["tango_partial_derivatives", "transition_3D"],
        lambda: event_calc.tango_partial_derivatives[0][transition_3D],
    )
    index += 1

    d_d_cos = compute_value(
        "d_d_cos",
        ["tango_partial_derivatives", "transition_3D"],
        lambda: event_calc.tango_partial_derivatives[1][transition_3D],
    )
    index += 1

    res_sum_loc_sigma = compute_value(
        "res_sum_loc_sigma",
        ["d_de", "Ns", "cos_err_perm", "d_d_cos"],
        lambda: res_sum_loc_sigma_func(d_de, Ns, cos_err_perm, d_d_cos, eres),
    )
    index += 1

    res_sum_loc_v = compute_value(
        "res_sum_loc_v",
        ["res_sum_loc", "res_sum_loc_sigma"],
        lambda: res_sum_loc / res_sum_loc_sigma,
    )
    index += 1

    # if compute_mode and not njit_any(boolean_vector[index : index + 3]):
    #     index += 3
    #     res_loc_geo = None
    #     res_loc_geo_sigma = None
    #     res_loc_geo_v = None
    # else:
    res_loc_geo = compute_value(
        "res_loc_geo",
        ["local_tango_estimates_perm", "energies_perm", "scattered_energy"],
        lambda: res_loc_geo_func(
            local_tango_estimates_perm, energies_perm, scattered_energy
        ),
    )
    index += 1

    res_loc_geo_sigma = compute_value(
        "res_loc_geo_sigma",
        [
            "d_de",
            "scattered_energy",
            "energy_rev_cumsum",
            "Ns",
            "cos_err_perm",
            "d_d_cos",
        ],
        lambda: res_loc_geo_sigma_func(
            d_de, scattered_energy, energy_rev_cumsum, Ns, cos_err_perm, d_d_cos, eres
        ),
    )
    index += 1

    res_loc_geo_v = compute_value(
        "res_loc_geo_v",
        ["res_loc_geo", "res_loc_geo_sigma"],
        lambda: res_loc_geo / res_loc_geo_sigma,
    )
    index += 1

    res_cos = compute_value(
        "res_cos",
        ["cos_act_perm", "cos_theor_perm"],
        lambda: np.abs(cos_act_perm - cos_theor_perm),
    )
    index += 1

    res_cos_cap = compute_value(
        "res_cos_cap",
        ["cos_act_perm", "cos_theor_perm"],
        lambda: res_cos_cap_func(cos_act_perm, cos_theor_perm),
    )
    index += 1

    res_cos_sigma = compute_value(
        "res_cos_sigma",
        ["cos_err_perm", "cos_theor_sigma_perm"],
        lambda: res_cos_sigma_func(cos_err_perm, cos_theor_sigma_perm, eres),
    )
    index += 1

    res_cos_v = compute_value(
        "res_cos_v", ["res_cos", "res_cos_sigma"], lambda: res_cos / res_cos_sigma
    )
    index += 1

    res_cos_cap_v = compute_value(
        "res_cos_cap_v",
        ["res_cos_cap", "res_cos_sigma"],
        lambda: res_cos_cap / res_cos_sigma,
    )
    index += 1

    res_theta = compute_value(
        "res_theta",
        ["theta_act_perm", "theta_theor_perm"],
        lambda: res_theta_func(fix_nan, theta_act_perm, theta_theor_perm),
    )
    index += 1

    res_theta_cap = compute_value(
        "res_theta_cap",
        ["theta_act_perm", "cos_theor_perm"],
        lambda: res_theta_cap_func(fix_nan, theta_act_perm, cos_theor_perm),
    )
    index += 1

    res_theta_sigma = compute_value(
        "res_theta_sigma",
        ["theta_err_perm", "theta_theor_sigma_perm"],
        lambda: res_theta_sigma_func(theta_err_perm, theta_theor_sigma_perm, eres),
    )
    index += 1

    res_theta_v = compute_value(
        "res_theta_v",
        ["res_theta", "res_theta_sigma"],
        lambda: res_theta / res_theta_sigma,
    )
    index += 1

    res_theta_cap_v = compute_value(
        "res_theta_cap_v",
        ["res_theta_cap", "res_theta_sigma"],
        lambda: res_theta_cap / res_theta_sigma,
    )
    index += 1

    compton_penalty = compute_value(
        "compton_penalty",
        ["cos_theor_perm"],
        lambda: phys.compton_penalty(cos_theor_perm),
    )
    index += 1

    compton_penalty_ell1 = compute_value(
        "compton_penalty_ell1",
        ["cos_theor_perm"],
        lambda: phys.compton_penalty_ell1(cos_theor_perm),
    )
    index += 1

    linear_attenuation_abs = compute_value(
        "linear_attenuation_abs",
        ["energy_rev_cumsum"],
        lambda: phys.lin_att_abs(energy_rev_cumsum),
    )
    index += 1

    linear_attenuation_compt = compute_value(
        "linear_attenuation_compt",
        ["energy_rev_cumsum"],
        lambda: phys.lin_att_compt(energy_rev_cumsum),
    )
    index += 1

    linear_attenuation_pair = compute_value(
        "linear_attenuation_pair",
        ["energy_rev_cumsum"],
        lambda: phys.lin_att_pair(energy_rev_cumsum),
    )
    index += 1

    lin_mu_total = compute_value(
        "lin_mu_total",
        [
            "linear_attenuation_abs",
            "linear_attenuation_compt",
            "linear_attenuation_pair",
        ],
        lambda: linear_attenuation_abs
        + linear_attenuation_compt
        + linear_attenuation_pair,
    )
    index += 1

    klein_nishina_use_Ei = compute_value(
        "klein_nishina_use_Ei",
        ["energy_rev_cumsum", "cos_act_perm"],
        lambda: phys.KN_differential_cross(
            energy_rev_cumsum[:-1],
            1 - cos_act_perm,
            energy_rev_cumsum[1:],
            relative=False,
            integrate=True,
        ),
    )
    index += 1

    klein_nishina = compute_value(
        "klein_nishina",
        ["energy_rev_cumsum", "cos_act_perm"],
        lambda: phys.KN_differential_cross(
            energy_rev_cumsum[:-1], 1 - cos_act_perm, relative=False, integrate=True
        ),
    )
    index += 1

    # klein_nishina_differential_cross_section_use_Ei = compute_value(
    #     "klein_nishina_differential_cross_section_use_Ei", ["energy_rev_cumsum", "cos_act_perm"],
    #     lambda: phys.KN_differential_cross(energy_rev_cumsum[:-1], 1 - cos_act_perm, energy_rev_cumsum[1:], relative=False, integrate=False))
    # index += 1

    # klein_nishina_differential_cross_section = compute_value(
    #     "klein_nishina_differential_cross_section", ["energy_rev_cumsum", "cos_act_perm"],
    #     lambda: phys.KN_differential_cross(energy_rev_cumsum[:-1], 1 - cos_act_perm, relative=False, integrate=False))
    # index += 1

    klein_nishina_relative_use_Ei = compute_value(
        "klein_nishina_relative_use_Ei",
        ["energy_rev_cumsum", "cos_act_perm", "linear_attenuation_compt"],
        lambda: phys.KN_differential_cross(
            energy_rev_cumsum[:-1],
            1 - cos_act_perm,
            energy_rev_cumsum[1:],
            linear_attenuation_compt / phys.RANGE_PROCESS,
            relative=True,
            integrate=True,
        ),
    )
    index += 1

    klein_nishina_relative = compute_value(
        "klein_nishina_relative",
        ["energy_rev_cumsum", "cos_act_perm", "linear_attenuation_compt"],
        lambda: phys.KN_differential_cross(
            energy_rev_cumsum[:-1],
            1 - cos_act_perm,
            sigma_compt=linear_attenuation_compt / phys.RANGE_PROCESS,
            relative=True,
            integrate=True,
        ),
    )
    index += 1

    # klein_nishina_differential_cross_section_relative_use_Ei = compute_value(
    #     "klein_nishina_differential_cross_section_relative_use_Ei", ["energy_rev_cumsum", "cos_act_perm", "linear_attenuation_compt"],
    #     lambda: phys.KN_differential_cross(energy_rev_cumsum[:-1], 1 - cos_act_perm, energy_rev_cumsum[1:], linear_attenuation_compt/phys.RANGE_PROCESS, relative = True, integrate=False))
    # index += 1

    # klein_nishina_differential_cross_section_relative = compute_value(
    #     "klein_nishina_differential_cross_section_relative", ["energy_rev_cumsum", "cos_act_perm", "linear_attenuation_compt"],
    #     lambda: phys.KN_differential_cross(energy_rev_cumsum[:-1], 1 - cos_act_perm, sigma_compt=linear_attenuation_compt/phys.RANGE_PROCESS, relative=True, integrate=False))
    # index += 1

    radii_perm = compute_value(
        "radii_perm", ["radii"], lambda: event_calc.radii[transition_1D[1:]]
    )
    index += 1

    escape_probability = compute_value(
        "escape_probability",
        ["energies_perm", "energy_sum"],
        lambda: ft_escape_probability(
            event_calc.point_matrix[permutation[-2]],
            event_calc.point_matrix[permutation[-1]],
            energies_perm[-1],
            start_energy - energy_sum,
            outer_radius,
        ),
    )
    index += 1

    if name_mode:
        return names
    if dependency_mode:
        return dependencies_dict
    return perm_level_values(
        # transition_1D,
        transition_2D,
        transition_3D,
        energies_perm,
        energy_sum,
        # energy_cumsum,
        energy_cumsum_with_zero,
        energy_rev_cumsum,
        Ns,
        # energy_rev_cumsum_sigma,
        distance_perm,
        ge_distance_perm,
        cos_act_perm,
        cos_err_perm,
        theta_act_perm,
        theta_err_perm,
        local_tango_estimates_perm,
        tango_estimates_perm,
        tango_estimates_sigma_perm,
        # estimate_start_energy_perm,
        # sqrt_N,
        # estimate_start_energy_sigma_weighted_perm,
        cos_theor_perm,
        cos_theor_sigma_perm,
        theta_theor_perm,
        theta_theor_sigma_perm,
        scattered_energy,
        # scattered_energy_sigma,
        res_sum_geo,
        res_sum_geo_sigma,
        res_sum_geo_v,
        res_sum_loc,
        d_de,
        d_d_cos,
        res_sum_loc_sigma,
        res_sum_loc_v,
        res_loc_geo,
        res_loc_geo_sigma,
        res_loc_geo_v,
        res_cos,
        res_cos_cap,
        res_cos_sigma,
        res_cos_v,
        res_cos_cap_v,
        res_theta,
        res_theta_cap,
        res_theta_sigma,
        res_theta_v,
        res_theta_cap_v,
        compton_penalty,
        compton_penalty_ell1,
        linear_attenuation_abs,
        linear_attenuation_compt,
        linear_attenuation_pair,
        lin_mu_total,
        klein_nishina_use_Ei,
        klein_nishina,
        # klein_nishina_differential_cross_section_use_Ei,
        # klein_nishina_differential_cross_section,
        klein_nishina_relative_use_Ei,
        klein_nishina_relative,
        # klein_nishina_differential_cross_section_relative_use_Ei,
        # klein_nishina_differential_cross_section_relative,
        radii_perm,
        escape_probability,
    )


# %%
# fmt: off
# def perm_atoms_explicit(
#     permutation:tuple[int],
#     start_point:int = 0,
#     start_energy:float = None,
#     point_matrix:Optional[np.ndarray] = None,
#     energy_matrix:Optional[np.ndarray] = None,
#     # position_uncertainty:Optional[np.ndarray] = None,
#     # energy_uncertainty:Optional[np.ndarray] = None,
#     distance:Optional[np.ndarray] = None,
#     # angle_distance:Optional[np.ndarray] = None,
#     ge_distance:Optional[np.ndarray] = None,
#     cos_act:Optional[np.ndarray] = None,
#     cos_err:Optional[np.ndarray] = None,
#     # theta_err:Optional[np.ndarray] = None,
#     tango_estimates:Optional[np.ndarray] = None,
#     tango_partial_derivatives:Optional[Tuple[np.ndarray]] = None,
#     tango_estimates_sigma:Optional[np.ndarray] = None,
#     use_threshold:bool = False,
#     Nmi:Optional[int] = None,
#     fix_nan:float = 2*np.pi,
#     computation_bool:np.ndarray[bool] = None,
#     all_computations:bool = False,
#     eres:float = 1e-3,
# ):
#     """
#     Inputs are general computational atoms
    
#     Outputs are permuted computational atoms
#     """
#     if Nmi is None:
#         Nmi = len(permutation)
#     if start_point is not None:
#         permutation = tuple([start_point] + list(permutation))  # include the start point; 0
#     transition_1D = np.array(permutation)
#     transition_2D = perm_to_transition(permutation, D=2)  # transition indices for array lookups
#     transition_3D = perm_to_transition(permutation, D=3)  # transition indices for array lookups

#     energies_perm = energy_matrix[transition_1D[1:]]
#     energy_sum = np.sum(energies_perm)
#     if start_energy is None:  # start_energy != energy_sum if only a partial permutation
#         start_energy = energy_sum
#     energy_cumsum = cumsum(energies_perm)
#     energy_cumsum_with_zero = cumsum(energy_matrix[transition_1D])
#     energy_rev_cumsum = reverse_cumsum(energies_perm) + start_energy - energy_sum
#     Ns = np.arange(Nmi, Nmi - len(permutation), -1, dtype=int)
#     energy_rev_cumsum_sigma = eres * np.sqrt(Ns[:-1])

#     distance_perm = distance[transition_2D]
#     ge_distance_perm = ge_distance[transition_2D]
#     cos_act_perm = cos_act[transition_3D]
#     cos_err_perm = cos_err[transition_3D]
#     theta_act_perm = np.arccos(cos_act[transition_3D])
#     theta_err_perm = cos_err[transition_3D] / np.sqrt(1.0 - cos_act[transition_3D]**2)
#     local_tango_estimates_perm = tango_estimates[transition_3D]
#     tango_estimates_sigma_perm = tango_estimates_sigma[transition_3D]

#     # estimate_start_energy_perm = np.mean(energy_cumsum[:-2] + local_tango_estimates_perm)
#     estimate_start_energy_perm = np.mean(energy_cumsum_with_zero[:-2] + tango_estimates[transition_3D])
#     if use_threshold:
#         if estimate_start_energy_perm < energy_sum or phys.njit_cos_theor(estimate_start_energy_perm - energy_sum + energies_perm[-1], estimate_start_energy_perm - energy_sum) < -1:
#             estimate_start_energy_perm = energy_sum

#     sqrt_N = np.sqrt(np.arange(1, len(energies_perm), 1))
#     estimate_start_energy_sigma_weighted_perm = np.sum((energy_cumsum_with_zero[:-2] +  local_tango_estimates_perm)/(tango_estimates_sigma_perm + eres * sqrt_N))/np.sum(1.0 / (tango_estimates_sigma_perm + eres * sqrt_N))
#     if use_threshold:
#         if estimate_start_energy_sigma_weighted_perm < energy_sum or phys.njit_cos_theor(estimate_start_energy_sigma_weighted_perm - energy_sum + energies_perm[-1], estimate_start_energy_sigma_weighted_perm - energy_sum) < -1:
#             estimate_start_energy_sigma_weighted_perm = energy_sum

#     cos_theor_perm = phys.cos_theor_sequence(energy_rev_cumsum)
#     cos_theor_sigma_perm = phys.cos_theor_sigma(energy_rev_cumsum[:-1], energy_rev_cumsum[1:], Nmi - 1)

#     theta_theor_perm = phys.theta_theor(energy_rev_cumsum[:-1], energy_rev_cumsum[1:])
#     theta_theor_sigma_perm = cos_theor_sigma_perm / np.sqrt( np.abs(1 - cos_theor_perm**2) )

#     scattered_energy = phys.outgoing_energy_csf(energy_rev_cumsum[:-1], 1 - cos_act_perm)
#     scattered_energy_sigma = phys.outgoing_energy_csf_sigma(energy_rev_cumsum[:-1], 1 - cos_act_perm, cos_err_perm, energy_rev_cumsum_sigma[1:])

#     res_sum_geo = energy_rev_cumsum[1:] - scattered_energy
#     res_sum_geo_sigma = np.sqrt(
#         eres**2 * (
#             (scattered_energy / energy_rev_cumsum[:-1])**4 + Ns[:-2] * (1 - (scattered_energy / energy_rev_cumsum[:-1])**2)**2
#             ) + (cos_err_perm * (scattered_energy**2 / phys.MEC2))**2
#         )

#     res_sum_loc = energy_rev_cumsum[:-1] - local_tango_estimates_perm

#     d_de = tango_partial_derivatives[0][transition_3D]
#     d_d_cos = tango_partial_derivatives[1][transition_3D]

#     res_sum_loc_sigma = np.sqrt((eres**2 * ((1 - d_de) ** 2 + Ns[:-2] - 1)) + (cos_err_perm * (d_d_cos)) ** 2)

#     res_loc_geo = (local_tango_estimates_perm - energies_perm[:-1]) - scattered_energy
#     res_loc_geo_sigma = np.sqrt(
#             eres**2
#             * ((d_de - (scattered_energy / energy_rev_cumsum[1:]) ** 2) ** 2 + Ns[:-2] * (scattered_energy / energy_rev_cumsum[1:]) ** 4)
#             + cos_err_perm**2 * (d_d_cos - scattered_energy**2 / phys.MEC2) ** 2
#         )
#     res_cos = cos_act_perm - cos_theor_perm
#     res_cos_cap = cos_act_perm - np.maximum(cos_theor_perm, -1)
#     res_cos_sigma = np.sqrt(cos_err_perm**2 + (eres * cos_theor_sigma_perm)**2)
#     res_theta = theta_act_perm - theta_theor_perm
#     if fix_nan is not None:
#         res_theta[np.isnan(res_theta)] = fix_nan
#     res_theta_cap = theta_act_perm - np.arccos(np.maximum(cos_theor_perm, -1))
#     if fix_nan is not None:
#         res_theta_cap[np.isnan(res_theta_cap)] = fix_nan
#     res_theta_sigma = np.sqrt(theta_err_perm**2 + (eres * theta_theor_sigma_perm)**2)
#     compton_penalty = phys.compton_penalty(cos_theor_perm)
#     compton_penalty_ell1 = phys.compton_penalty_ell1(cos_theor_perm)

#     linear_attenuation_abs = phys.lin_att_abs(energy_rev_cumsum)
#     linear_attenuation_compt = phys.lin_att_compt(energy_rev_cumsum)
#     linear_attenuation_pair = phys.lin_att_pair(energy_rev_cumsum)
#     lin_mu_total = linear_attenuation_abs + linear_attenuation_compt + linear_attenuation_pair

#     klein_nishina_use_Ei = phys.KN_differential_cross(energy_rev_cumsum[:-1], 1 - cos_act_perm, energy_rev_cumsum[1:], relative = False, integrate=True)
#     klein_nishina = phys.KN_differential_cross(energy_rev_cumsum[:-1], 1 - cos_act_perm, relative = False, integrate=True)
#     klein_nishina_differential_cross_section_use_Ei = phys.KN_differential_cross(energy_rev_cumsum[:-1], 1 - cos_act_perm, energy_rev_cumsum[1:], integrate=False)
#     klein_nishina_differential_cross_section = phys.KN_differential_cross(energy_rev_cumsum[:-1], 1 - cos_act_perm, integrate=False)

#     klein_nishina_relative_use_Ei = phys.KN_differential_cross(energy_rev_cumsum[:-1], 1 - cos_act_perm, energy_rev_cumsum[1:], relative = True, integrate=True)
#     klein_nishina_relative = phys.KN_differential_cross(energy_rev_cumsum[:-1], 1 - cos_act_perm, relative = True, integrate=True)
#     klein_nishina_differential_cross_section_relative_use_Ei = phys.KN_differential_cross(energy_rev_cumsum[:-1], 1 - cos_act_perm, energy_rev_cumsum[1:], relative = True, integrate=False)
#     klein_nishina_differential_cross_section_relative = phys.KN_differential_cross(energy_rev_cumsum[:-1], 1 - cos_act_perm, relative=True, integrate=False)

# fmt: on
