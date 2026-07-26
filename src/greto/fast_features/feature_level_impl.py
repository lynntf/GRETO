"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Feature level computations
"""

from __future__ import annotations

from typing import Iterable, Optional

import numba
import numpy as np

from greto.fast_features.permutation_level import perm_level_values
from greto.physics import RANGE_PROCESS
from greto.utils import njit_any, njit_max, njit_mean, njit_min, njit_norm, njit_sum


@numba.njit
def rc_wmean_1v_penalty_removed_func(compton_penalty, res_cos_v, res_cos_sigma):
    """Need to check if all values are removed by penalty"""
    if not (max(compton_penalty.shape) - njit_sum(compton_penalty)) < 1:
        return njit_sum(res_cos_v * (1.0 - compton_penalty)) / njit_sum(
            1.0 / res_cos_sigma * (1.0 - compton_penalty)
        )
    return 0.0


@numba.njit
def rc_wmean_2v_penalty_removed_func(compton_penalty, res_cos_v, res_cos_sigma):
    """Need to check if all values are removed by penalty"""
    if not (max(compton_penalty.shape) - njit_sum(compton_penalty)) < 1:
        return njit_sum(res_cos_v**2 * (1.0 - compton_penalty)) / njit_sum(
            (1.0 / res_cos_sigma * (1.0 - compton_penalty)) ** 2
        )
    return 0.0


@numba.njit
def rth_wmean_1v_penalty_removed_func(compton_penalty, res_theta_v, res_theta_sigma):
    if not (max(compton_penalty.shape) - njit_sum(compton_penalty)) < 1:  # zeroed:
        denom = njit_sum((1 / res_theta_sigma) * (1.0 - compton_penalty))
        if denom > 0:
            return njit_sum(res_theta_v * (1.0 - compton_penalty)) / denom
    return 0.0


@numba.njit
def rth_wmean_2v_penalty_removed_func(compton_penalty, res_theta_v, res_theta_sigma):
    if not (max(compton_penalty.shape) - njit_sum(compton_penalty)) < 1:  # zeroed:
        denom = njit_sum(((1 / res_theta_sigma) * (1.0 - compton_penalty)) ** 2)
        if denom > 0:
            return njit_sum(res_theta_v**2 * (1.0 - compton_penalty)) / denom
    return 0.0


@numba.njit
def wmean_1v_func(stdev_weighted_value, stdev):
    return njit_sum(stdev_weighted_value) / njit_sum(1 / stdev)


@numba.njit
def wmean_2v_func(stdev_weighted_value, stdev):
    return njit_sum(stdev_weighted_value**2) / njit_sum(1 / stdev**2)


def feature_values(
    permutation: Iterable[int],
    perm_calc: perm_level_values,
    Nmi: Optional[int] = None,
    boolean_vector: Optional[Iterable[bool]] = None,
    name_mode: bool = False,
    dependency_mode: bool = False,
    all_computations: bool = False,
    number_of_values: int = 240,
):
    """
    Feature level values

    Given permutation measurements, get values the values for features.
    """
    compute_mode = not name_mode and not dependency_mode

    if compute_mode:
        if Nmi is None:
            Nmi = len(permutation)
        if len(permutation) == 1:
            return np.zeros((number_of_values,), dtype=float)

    if compute_mode and boolean_vector is None:
        all_computations = True

    features_vector = np.zeros((number_of_values,), dtype=float)
    if all_computations:
        boolean_vector = np.ones(features_vector.shape, dtype=np.bool_)
    elif compute_mode and boolean_vector is not None:
        boolean_vector = np.asarray(boolean_vector, dtype=np.bool_)

    names = []
    dependencies_dict = {}
    index = 0

    def emit_feature(name, dependencies, compute_fn):
        nonlocal index
        if compute_mode:
            if boolean_vector[index]:
                features_vector[index] = compute_fn()
        elif name_mode:
            names.append(name)
        elif dependency_mode:
            dependencies_dict[name] = dependencies
        index += 1

    def emit_feature_block(block_features, block_size):
        nonlocal index
        if compute_mode and block_size and not njit_any(boolean_vector[index : index + block_size]):
            index += block_size
            return
        for name, dependencies, compute_fn in block_features:
            emit_feature(name, dependencies, compute_fn)

    def emit_feature_group(blocks):
        nonlocal index
        total_size = sum(block_size for block_size, _ in blocks)
        if compute_mode and total_size and not njit_any(boolean_vector[index : index + total_size]):
            index += total_size
            return
        for block_size, block_features in blocks:
            emit_feature_block(block_features, block_size)

    feature_groups = [
        (
            46,
            [
                (
                    22,
                    [
                        ("rsg_sum_1", ["res_sum_geo"], lambda: njit_sum(perm_calc.res_sum_geo)),
                        ("rsg_sum_1_first", ["res_sum_geo"], lambda: perm_calc.res_sum_geo[0]),
                        ("rsg_mean_1", ["res_sum_geo"], lambda: njit_mean(perm_calc.res_sum_geo)),
                        (
                            "rsg_mean_1_first",
                            ["res_sum_geo"],
                            lambda: perm_calc.res_sum_geo[0] / len(perm_calc.res_sum_geo),
                        ),
                        (
                            "rsg_wmean_1v",
                            ["res_sum_geo_v", "res_sum_geo_sigma"],
                            lambda: wmean_1v_func(perm_calc.res_sum_geo_v, perm_calc.res_sum_geo_sigma),
                        ),
                        (
                            "rsg_wmean_1v_first",
                            ["res_sum_geo_v", "res_sum_geo_sigma"],
                            lambda: perm_calc.res_sum_geo_v[0]
                            / (1.0 / perm_calc.res_sum_geo_sigma[0]),
                        ),
                        ("rsg_norm_2", ["res_sum_geo"], lambda: njit_norm(perm_calc.res_sum_geo) / Nmi),
                        ("rsg_sum_2", ["res_sum_geo"], lambda: njit_sum(perm_calc.res_sum_geo**2)),
                        ("rsg_sum_2_first", ["res_sum_geo"], lambda: perm_calc.res_sum_geo[0] ** 2),
                        ("rsg_mean_2", ["res_sum_geo"], lambda: njit_mean(perm_calc.res_sum_geo**2)),
                        (
                            "rsg_mean_2_first",
                            ["res_sum_geo"],
                            lambda: perm_calc.res_sum_geo[0] ** 2 / len(perm_calc.res_sum_geo),
                        ),
                        (
                            "rsg_wmean_2v",
                            ["res_sum_geo_v", "res_sum_geo_sigma"],
                            lambda: wmean_2v_func(perm_calc.res_sum_geo_v, perm_calc.res_sum_geo_sigma),
                        ),
                        (
                            "rsg_wmean_2v_first",
                            ["res_sum_geo_v", "res_sum_geo_sigma"],
                            lambda: perm_calc.res_sum_geo_v[0] ** 2
                            / ((1.0 / perm_calc.res_sum_geo_sigma[0]) ** 2),
                        ),
                        ("rsg_sum_1v", ["res_sum_geo_v"], lambda: njit_sum(perm_calc.res_sum_geo_v)),
                        ("rsg_sum_1v_first", ["res_sum_geo_v"], lambda: perm_calc.res_sum_geo_v[0]),
                        ("rsg_mean_1v", ["res_sum_geo_v"], lambda: njit_mean(perm_calc.res_sum_geo_v)),
                        (
                            "rsg_mean_1v_first",
                            ["res_sum_geo_v"],
                            lambda: perm_calc.res_sum_geo_v[0] / len(perm_calc.res_sum_geo_v),
                        ),
                        ("rsg_norm_2v", ["res_sum_geo_v"], lambda: njit_norm(perm_calc.res_sum_geo_v) / Nmi),
                        ("rsg_sum_2v", ["res_sum_geo_v"], lambda: njit_sum(perm_calc.res_sum_geo_v**2)),
                        ("rsg_sum_2v_first", ["res_sum_geo_v"], lambda: perm_calc.res_sum_geo_v[0] ** 2),
                        ("rsg_mean_2v", ["res_sum_geo_v"], lambda: njit_mean(perm_calc.res_sum_geo_v**2)),
                        (
                            "rsg_mean_2v_first",
                            ["res_sum_geo_v"],
                            lambda: perm_calc.res_sum_geo_v[0] ** 2 / len(perm_calc.res_sum_geo_v),
                        ),
                    ],
                ),
                (
                    12,
                    [
                        ("rsl_mean_1", ["res_sum_loc"], lambda: njit_mean(perm_calc.res_sum_loc)),
                        ("rsl_sum_1", ["res_sum_loc"], lambda: njit_sum(perm_calc.res_sum_loc)),
                        ("rsl_norm_2", ["res_sum_loc"], lambda: njit_norm(perm_calc.res_sum_loc) / Nmi),
                        ("rsl_sum_2", ["res_sum_loc"], lambda: njit_sum(perm_calc.res_sum_loc**2)),
                        ("rsl_mean_2", ["res_sum_loc"], lambda: njit_mean(perm_calc.res_sum_loc**2)),
                        ("rsl_sum_1v", ["res_sum_loc_v"], lambda: njit_sum(perm_calc.res_sum_loc_v)),
                        ("rsl_mean_1v", ["res_sum_loc_v"], lambda: njit_mean(perm_calc.res_sum_loc_v)),
                        (
                            "rsl_norm_2v",
                            ["res_sum_loc_v"],
                            lambda: njit_norm(perm_calc.res_sum_loc_v) / Nmi / np.sqrt(len(perm_calc.res_sum_loc_v)),
                        ),
                        ("rsl_mean_2v", ["res_sum_loc_v"], lambda: njit_mean(perm_calc.res_sum_loc_v**2)),
                        ("rsl_sum_2v", ["res_sum_loc_v"], lambda: njit_sum(perm_calc.res_sum_loc_v**2)),
                        (
                            "rsl_wmean_2v",
                            ["res_sum_loc_v", "res_sum_loc_sigma"],
                            lambda: wmean_2v_func(perm_calc.res_sum_loc_v, perm_calc.res_sum_loc_sigma),
                        ),
                        (
                            "rsl_wmean_1v",
                            ["res_sum_loc_v", "res_sum_loc_sigma"],
                            lambda: wmean_1v_func(perm_calc.res_sum_loc_v, perm_calc.res_sum_loc_sigma),
                        ),
                    ],
                ),
                (
                    12,
                    [
                        ("rlg_sum_1v", ["res_loc_geo_v"], lambda: njit_sum(perm_calc.res_loc_geo_v)),
                        ("rlg_mean_1v", ["res_loc_geo_v"], lambda: njit_mean(perm_calc.res_loc_geo_v)),
                        (
                            "rlg_norm_2v",
                            ["res_loc_geo_v"],
                            lambda: njit_norm(perm_calc.res_loc_geo_v) / Nmi / np.sqrt(len(perm_calc.res_loc_geo_v)),
                        ),
                        ("rlg_sum_2v", ["res_loc_geo_v"], lambda: njit_sum(perm_calc.res_loc_geo_v**2)),
                        ("rlg_mean_2v", ["res_loc_geo_v"], lambda: njit_mean(perm_calc.res_loc_geo_v**2)),
                        ("rlg_sum_1", ["res_loc_geo"], lambda: njit_sum(perm_calc.res_loc_geo)),
                        ("rlg_mean_1", ["res_loc_geo"], lambda: njit_mean(perm_calc.res_loc_geo)),
                        ("rlg_norm_2", ["res_loc_geo"], lambda: njit_norm(perm_calc.res_loc_geo) / Nmi),
                        (
                            "rlg_wmean_1v",
                            ["res_loc_geo_v", "res_loc_geo_sigma"],
                            lambda: wmean_1v_func(perm_calc.res_loc_geo_v, perm_calc.res_loc_geo_sigma),
                        ),
                        ("rlg_sum_2", ["res_loc_geo"], lambda: njit_sum(perm_calc.res_loc_geo**2)),
                        ("rlg_mean_2", ["res_loc_geo"], lambda: njit_mean(perm_calc.res_loc_geo**2)),
                        (
                            "rlg_wmean_2v",
                            ["res_loc_geo_v", "res_loc_geo_sigma"],
                            lambda: wmean_2v_func(perm_calc.res_loc_geo_v, perm_calc.res_loc_geo_sigma),
                        ),
                    ],
                ),
            ],
        ),
        (
            74,
            [
                (
                    6,
                    [
                        ("c_penalty_sum_1", ["compton_penalty"], lambda: njit_sum(perm_calc.compton_penalty)),
                        ("c_penalty_mean_1", ["compton_penalty"], lambda: njit_mean(perm_calc.compton_penalty)),
                        ("c_penalty_ell_sum_1", ["compton_penalty_ell1"], lambda: njit_sum(perm_calc.compton_penalty_ell1)),
                        ("c_penalty_ell_mean_1", ["compton_penalty_ell1"], lambda: njit_mean(perm_calc.compton_penalty_ell1)),
                        ("c_penalty_ell_sum_2", ["compton_penalty_ell1"], lambda: njit_sum(perm_calc.compton_penalty_ell1**2)),
                        ("c_penalty_ell_mean_2", ["compton_penalty_ell1"], lambda: njit_mean(perm_calc.compton_penalty_ell1**2)),
                    ],
                ),
                (
                    22,
                    [
                        ("rc_sum_1", ["res_cos"], lambda: njit_sum(perm_calc.res_cos)),
                        ("rc_mean_1", ["res_cos"], lambda: njit_mean(perm_calc.res_cos)),
                        ("rc_norm_2", ["res_cos"], lambda: njit_norm(perm_calc.res_cos) / Nmi),
                        ("rc_sum_2", ["res_cos"], lambda: njit_sum(perm_calc.res_cos**2)),
                        ("rc_mean_2", ["res_cos"], lambda: njit_mean(perm_calc.res_cos**2)),
                        ("rc_sum_1_penalty_removed", ["res_cos", "compton_penalty"], lambda: njit_sum(perm_calc.res_cos * (1.0 - perm_calc.compton_penalty))),
                        ("rc_mean_1_penalty_removed", ["res_cos", "compton_penalty"], lambda: njit_mean(perm_calc.res_cos * (1.0 - perm_calc.compton_penalty))),
                        ("rc_sum_2_penalty_removed", ["res_cos", "compton_penalty"], lambda: njit_sum(perm_calc.res_cos**2 * (1.0 - perm_calc.compton_penalty))),
                        ("rc_mean_2_penalty_removed", ["res_cos", "compton_penalty"], lambda: njit_mean(perm_calc.res_cos**2 * (1.0 - perm_calc.compton_penalty))),
                        ("rc_wmean_1v", ["res_cos_v", "res_cos_sigma"], lambda: wmean_1v_func(perm_calc.res_cos_v, perm_calc.res_cos_sigma)),
                        ("rc_wmean_2v", ["res_cos_v", "res_cos_sigma"], lambda: wmean_2v_func(perm_calc.res_cos_v, perm_calc.res_cos_sigma)),
                        ("rc_sum_1v", ["res_cos_v"], lambda: njit_sum(perm_calc.res_cos_v)),
                        ("rc_mean_1v", ["res_cos_v"], lambda: njit_mean(perm_calc.res_cos_v)),
                        ("rc_norm_2v", ["res_cos_v"], lambda: njit_norm(perm_calc.res_cos_v) / Nmi),
                        ("rc_sum_2v", ["res_cos_v"], lambda: njit_sum(perm_calc.res_cos_v**2)),
                        ("rc_mean_2v", ["res_cos_v"], lambda: njit_mean(perm_calc.res_cos_v**2)),
                        (
                            "rc_wmean_1v_penalty_removed",
                            ["compton_penalty", "res_cos_v", "res_cos_sigma"],
                            lambda: rc_wmean_1v_penalty_removed_func(
                                perm_calc.compton_penalty,
                                perm_calc.res_cos_v,
                                perm_calc.res_cos_sigma,
                            ),
                        ),
                        (
                            "rc_wmean_2v_penalty_removed",
                            ["compton_penalty", "res_cos_v", "res_cos_sigma"],
                            lambda: rc_wmean_2v_penalty_removed_func(
                                perm_calc.compton_penalty,
                                perm_calc.res_cos_v,
                                perm_calc.res_cos_sigma,
                            ),
                        ),
                        ("rc_sum_1v_penalty_removed", ["res_cos_v", "compton_penalty"], lambda: njit_sum(perm_calc.res_cos_v * (1.0 - perm_calc.compton_penalty))),
                        ("rc_mean_1v_penalty_removed", ["res_cos_v", "compton_penalty"], lambda: njit_mean(perm_calc.res_cos_v * (1.0 - perm_calc.compton_penalty))),
                        ("rc_sum_2v_penalty_removed", ["res_cos_v", "compton_penalty"], lambda: njit_sum(perm_calc.res_cos_v**2 * (1.0 - perm_calc.compton_penalty))),
                        ("rc_mean_2v_penalty_removed", ["res_cos_v", "compton_penalty"], lambda: njit_mean(perm_calc.res_cos_v**2 * (1.0 - perm_calc.compton_penalty))),
                    ],
                ),
                (
                    12,
                    [
                        ("rc_cap_sum_1", ["res_cos_cap"], lambda: njit_sum(perm_calc.res_cos_cap)),
                        ("rc_cap_mean_1", ["res_cos_cap"], lambda: njit_mean(perm_calc.res_cos_cap)),
                        ("rc_cap_norm_2", ["res_cos_cap"], lambda: njit_norm(perm_calc.res_cos_cap) / Nmi),
                        ("rc_cap_sum_2", ["res_cos_cap"], lambda: njit_sum(perm_calc.res_cos_cap**2)),
                        ("rc_cap_mean_2", ["res_cos_cap"], lambda: njit_mean(perm_calc.res_cos_cap**2)),
                        ("rc_cap_wmean_1v", ["res_cos_cap_v", "res_cos_sigma"], lambda: wmean_1v_func(perm_calc.res_cos_cap_v, perm_calc.res_cos_sigma)),
                        ("rc_cap_wmean_2v", ["res_cos_cap_v", "res_cos_sigma"], lambda: wmean_2v_func(perm_calc.res_cos_cap_v, perm_calc.res_cos_sigma)),
                        ("rc_cap_sum_1v", ["res_cos_cap_v"], lambda: njit_sum(perm_calc.res_cos_cap_v)),
                        ("rc_cap_mean_1v", ["res_cos_cap_v"], lambda: njit_mean(perm_calc.res_cos_cap_v)),
                        ("rc_cap_norm_2v", ["res_cos_cap_v"], lambda: njit_norm(perm_calc.res_cos_cap_v) / Nmi),
                        ("rc_cap_sum_2v", ["res_cos_cap_v"], lambda: njit_sum(perm_calc.res_cos_cap_v**2)),
                        ("rc_cap_mean_2v", ["res_cos_cap_v"], lambda: njit_mean(perm_calc.res_cos_cap_v**2)),
                    ],
                ),
                (
                    22,
                    [
                        ("rth_sum_1", ["res_theta"], lambda: njit_sum(perm_calc.res_theta)),
                        ("rth_mean_1", ["res_theta"], lambda: njit_mean(perm_calc.res_theta)),
                        ("rth_norm_2", ["res_theta"], lambda: njit_norm(perm_calc.res_theta) / Nmi),
                        ("rth_sum_2", ["res_theta"], lambda: njit_sum(perm_calc.res_theta**2)),
                        ("rth_mean_2", ["res_theta"], lambda: njit_mean(perm_calc.res_theta**2)),
                        ("rth_sum_1_penalty_removed", ["res_theta", "compton_penalty"], lambda: njit_sum(perm_calc.res_theta * (1.0 - perm_calc.compton_penalty))),
                        ("rth_mean_1_penalty_removed", ["res_theta", "compton_penalty"], lambda: njit_mean(perm_calc.res_theta * (1.0 - perm_calc.compton_penalty))),
                        ("rth_sum_2_penalty_removed", ["res_theta", "compton_penalty"], lambda: njit_sum(perm_calc.res_theta**2 * (1.0 - perm_calc.compton_penalty))),
                        ("rth_mean_2_penalty_removed", ["res_theta", "compton_penalty"], lambda: njit_mean(perm_calc.res_theta**2 * (1.0 - perm_calc.compton_penalty))),
                        ("rth_wmean_1v", ["res_theta_v", "res_theta_sigma"], lambda: wmean_1v_func(perm_calc.res_theta_v, perm_calc.res_theta_sigma)),
                        ("rth_wmean_2v", ["res_theta_v", "res_theta_sigma"], lambda: wmean_2v_func(perm_calc.res_theta_v, perm_calc.res_theta_sigma)),
                        ("rth_sum_1v", ["res_theta_v"], lambda: njit_sum(perm_calc.res_theta_v)),
                        ("rth_mean_1v", ["res_theta_v"], lambda: njit_mean(perm_calc.res_theta_v)),
                        ("rth_norm_2v", ["res_theta_v"], lambda: njit_norm(perm_calc.res_theta_v) / Nmi),
                        ("rth_sum_2v", ["res_theta_v"], lambda: njit_sum(perm_calc.res_theta_v**2)),
                        ("rth_mean_2v", ["res_theta_v"], lambda: njit_mean(perm_calc.res_theta_v**2)),
                        ("rth_sum_1v_penalty_removed", ["res_theta_v", "compton_penalty"], lambda: njit_sum(perm_calc.res_theta_v * (1.0 - perm_calc.compton_penalty))),
                        ("rth_mean_1v_penalty_removed", ["res_theta_v", "compton_penalty"], lambda: njit_mean(perm_calc.res_theta_v * (1.0 - perm_calc.compton_penalty))),
                        ("rth_sum_2v_penalty_removed", ["res_theta_v", "compton_penalty"], lambda: njit_sum(perm_calc.res_theta_v**2 * (1.0 - perm_calc.compton_penalty))),
                        ("rth_mean_2v_penalty_removed", ["res_theta_v", "compton_penalty"], lambda: njit_mean(perm_calc.res_theta_v**2 * (1.0 - perm_calc.compton_penalty))),
                        (
                            "rth_wmean_1v_penalty_removed",
                            ["compton_penalty", "res_theta_v", "res_theta_sigma"],
                            lambda: rth_wmean_1v_penalty_removed_func(
                                perm_calc.compton_penalty,
                                perm_calc.res_theta_v,
                                perm_calc.res_theta_sigma,
                            ),
                        ),
                        (
                            "rth_wmean_2v_penalty_removed",
                            ["compton_penalty", "res_theta_v", "res_theta_sigma"],
                            lambda: rth_wmean_2v_penalty_removed_func(
                                perm_calc.compton_penalty,
                                perm_calc.res_theta_v,
                                perm_calc.res_theta_sigma,
                            ),
                        ),
                    ],
                ),
                (
                    12,
                    [
                        ("rth_cap_sum_1", ["res_theta_cap"], lambda: njit_sum(perm_calc.res_theta_cap)),
                        ("rth_cap_mean_1", ["res_theta_cap"], lambda: njit_mean(perm_calc.res_theta_cap)),
                        ("rth_cap_norm_2", ["res_theta_cap"], lambda: njit_norm(perm_calc.res_theta_cap) / Nmi),
                        ("rth_cap_sum_2", ["res_theta_cap"], lambda: njit_sum(perm_calc.res_theta_cap**2)),
                        ("rth_cap_mean_2", ["res_theta_cap"], lambda: njit_mean(perm_calc.res_theta_cap**2)),
                        ("rth_cap_wmean_1v", ["res_theta_cap_v", "res_theta_sigma"], lambda: wmean_1v_func(perm_calc.res_theta_cap_v, perm_calc.res_theta_sigma)),
                        ("rth_cap_wmean_2v", ["res_theta_cap_v", "res_theta_sigma"], lambda: wmean_2v_func(perm_calc.res_theta_cap_v, perm_calc.res_theta_sigma)),
                        ("rth_cap_sum_1v", ["res_theta_cap_v"], lambda: njit_sum(perm_calc.res_theta_cap_v)),
                        ("rth_cap_mean_1v", ["res_theta_cap_v"], lambda: njit_mean(perm_calc.res_theta_cap_v)),
                        ("rth_cap_norm_2v", ["res_theta_cap_v"], lambda: njit_norm(perm_calc.res_theta_cap_v) / Nmi),
                        ("rth_cap_sum_2v", ["res_theta_cap_v"], lambda: njit_sum(perm_calc.res_theta_cap_v**2)),
                        ("rth_cap_mean_2v", ["res_theta_cap_v"], lambda: njit_mean(perm_calc.res_theta_cap_v**2)),
                    ],
                ),
            ],
        ),
        (
            40,
            [
                (
                    4,
                    [
                        ("distances_sum", ["distance_perm"], lambda: njit_sum(perm_calc.distance_perm)),
                        ("distances_mean", ["distance_perm"], lambda: njit_mean(perm_calc.distance_perm)),
                        ("ge_distances_sum", ["ge_distance_perm"], lambda: njit_sum(perm_calc.ge_distance_perm)),
                        ("ge_distances_mean", ["ge_distance_perm"], lambda: njit_mean(perm_calc.ge_distance_perm)),
                    ],
                ),
                (
                    15,
                    [
                        ("cross_abs_sum", ["linear_attenuation_abs"], lambda: njit_sum(perm_calc.linear_attenuation_abs)),
                        ("cross_abs_final", ["linear_attenuation_abs"], lambda: perm_calc.linear_attenuation_abs[-1]),
                        ("cross_abs_mean", ["linear_attenuation_abs"], lambda: njit_mean(perm_calc.linear_attenuation_abs)),
                        ("cross_abs_max", ["linear_attenuation_abs"], lambda: njit_max(perm_calc.linear_attenuation_abs)),
                        ("cross_abs_ge_dist_sum", ["linear_attenuation_abs", "ge_distance_perm"], lambda: njit_sum(perm_calc.linear_attenuation_abs * perm_calc.ge_distance_perm)),
                        ("cross_abs_ge_dist_final", ["linear_attenuation_abs", "ge_distance_perm"], lambda: perm_calc.linear_attenuation_abs[-1] * perm_calc.ge_distance_perm[-1]),
                        ("cross_abs_ge_dist_mean", ["linear_attenuation_abs", "ge_distance_perm"], lambda: njit_mean(perm_calc.linear_attenuation_abs * perm_calc.ge_distance_perm)),
                        ("cross_abs_ge_dist_max", ["linear_attenuation_abs", "ge_distance_perm"], lambda: njit_max(perm_calc.linear_attenuation_abs * perm_calc.ge_distance_perm)),
                        ("cross_abs_dist_sum", ["linear_attenuation_abs", "distance_perm"], lambda: njit_sum(perm_calc.linear_attenuation_abs * perm_calc.distance_perm)),
                        ("cross_abs_dist_final", ["linear_attenuation_abs", "distance_perm"], lambda: perm_calc.linear_attenuation_abs[-1] * perm_calc.distance_perm[-1]),
                        ("cross_abs_dist_mean", ["linear_attenuation_abs", "distance_perm"], lambda: njit_mean(perm_calc.linear_attenuation_abs * perm_calc.distance_perm)),
                        ("cross_abs_dist_max", ["linear_attenuation_abs", "distance_perm"], lambda: njit_max(perm_calc.linear_attenuation_abs * perm_calc.distance_perm)),
                        ("cross_abs_min", ["linear_attenuation_abs"], lambda: njit_min(perm_calc.linear_attenuation_abs)),
                        ("cross_abs_ge_dist_min", ["linear_attenuation_abs", "ge_distance_perm"], lambda: njit_min(perm_calc.linear_attenuation_abs * perm_calc.ge_distance_perm)),
                        ("cross_abs_dist_min", ["linear_attenuation_abs", "distance_perm"], lambda: njit_min(perm_calc.linear_attenuation_abs * perm_calc.distance_perm)),
                    ],
                ),
                (
                    21,
                    [
                        ("cross_compt_sum", ["linear_attenuation_compt"], lambda: njit_sum(perm_calc.linear_attenuation_compt)),
                        ("cross_compt_mean", ["linear_attenuation_compt"], lambda: njit_mean(perm_calc.linear_attenuation_compt)),
                        ("cross_compt_max", ["linear_attenuation_compt"], lambda: njit_max(perm_calc.linear_attenuation_compt)),
                        ("cross_compt_ge_dist_sum", ["linear_attenuation_compt", "ge_distance_perm"], lambda: njit_sum(perm_calc.linear_attenuation_compt * perm_calc.ge_distance_perm)),
                        ("cross_compt_ge_dist_mean", ["linear_attenuation_compt", "ge_distance_perm"], lambda: njit_mean(perm_calc.linear_attenuation_compt * perm_calc.ge_distance_perm)),
                        ("cross_compt_ge_dist_max", ["linear_attenuation_compt", "ge_distance_perm"], lambda: njit_max(perm_calc.linear_attenuation_compt * perm_calc.ge_distance_perm)),
                        ("cross_compt_dist_sum", ["linear_attenuation_compt", "distance_perm"], lambda: njit_sum(perm_calc.linear_attenuation_compt * perm_calc.distance_perm)),
                        ("cross_compt_dist_mean", ["linear_attenuation_compt", "distance_perm"], lambda: njit_mean(perm_calc.linear_attenuation_compt * perm_calc.distance_perm)),
                        ("cross_compt_dist_max", ["linear_attenuation_compt", "distance_perm"], lambda: njit_max(perm_calc.linear_attenuation_compt * perm_calc.distance_perm)),
                        ("cross_compt_min", ["linear_attenuation_compt"], lambda: njit_min(perm_calc.linear_attenuation_compt)),
                        ("cross_compt_ge_dist_min", ["linear_attenuation_compt", "ge_distance_perm"], lambda: njit_min(perm_calc.linear_attenuation_compt * perm_calc.ge_distance_perm)),
                        ("cross_compt_dist_min", ["linear_attenuation_compt", "distance_perm"], lambda: njit_min(perm_calc.linear_attenuation_compt * perm_calc.distance_perm)),
                        ("cross_compt_sum_nonfinal", ["linear_attenuation_compt"], lambda: njit_sum(perm_calc.linear_attenuation_compt[:-1])),
                        ("cross_compt_mean_nonfinal", ["linear_attenuation_compt"], lambda: njit_mean(perm_calc.linear_attenuation_compt[:-1])),
                        ("cross_compt_min_nonfinal", ["linear_attenuation_compt"], lambda: njit_min(perm_calc.linear_attenuation_compt[:-1])),
                        ("cross_compt_dist_sum_nonfinal", ["linear_attenuation_compt", "distance_perm"], lambda: njit_sum(perm_calc.linear_attenuation_compt[:-1] * perm_calc.distance_perm[:-1])),
                        ("cross_compt_dist_mean_nonfinal", ["linear_attenuation_compt", "distance_perm"], lambda: njit_mean(perm_calc.linear_attenuation_compt[:-1] * perm_calc.distance_perm[:-1])),
                        ("cross_compt_dist_min_nonfinal", ["linear_attenuation_compt", "distance_perm"], lambda: njit_min(perm_calc.linear_attenuation_compt[:-1] * perm_calc.distance_perm[:-1])),
                        ("cross_compt_ge_dist_sum_nonfinal", ["linear_attenuation_compt", "ge_distance_perm"], lambda: njit_sum(perm_calc.linear_attenuation_compt[:-1] * perm_calc.ge_distance_perm[:-1])),
                        ("cross_compt_ge_dist_mean_nonfinal", ["linear_attenuation_compt", "ge_distance_perm"], lambda: njit_mean(perm_calc.linear_attenuation_compt[:-1] * perm_calc.ge_distance_perm[:-1])),
                        ("cross_compt_ge_dist_min_nonfinal", ["linear_attenuation_compt", "ge_distance_perm"], lambda: njit_min(perm_calc.linear_attenuation_compt[:-1] * perm_calc.ge_distance_perm[:-1])),
                    ],
                ),
            ],
        ),
        (
            24,
            [
                ("p_abs_sum", ["linear_attenuation_abs", "lin_mu_total"], lambda: njit_sum(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total)),
                ("p_abs_final", ["linear_attenuation_abs", "lin_mu_total"], lambda: perm_calc.linear_attenuation_abs[-1] / perm_calc.lin_mu_total[-1]),
                ("p_abs_mean", ["linear_attenuation_abs", "lin_mu_total"], lambda: njit_mean(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total)),
                ("p_abs_max", ["linear_attenuation_abs", "lin_mu_total"], lambda: njit_max(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total)),
                ("p_abs_min", ["linear_attenuation_abs", "lin_mu_total"], lambda: njit_min(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total)),
                ("-log_p_abs_sum", ["linear_attenuation_abs", "lin_mu_total"], lambda: njit_sum(-np.log(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total))),
                ("-log_p_abs_final", ["linear_attenuation_abs", "lin_mu_total"], lambda: -np.log(perm_calc.linear_attenuation_abs[-1] / perm_calc.lin_mu_total[-1])),
                ("-log_p_abs_mean", ["linear_attenuation_abs", "lin_mu_total"], lambda: njit_mean(-np.log(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total))),
                ("-log_p_abs_max", ["linear_attenuation_abs", "lin_mu_total"], lambda: njit_max(-np.log(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total))),
                ("-log_p_abs_min", ["linear_attenuation_abs", "lin_mu_total"], lambda: njit_min(-np.log(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total))),
                ("p_compt_sum", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_sum(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total)),
                ("p_compt_mean", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_mean(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total)),
                ("p_compt_max", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_max(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total)),
                ("p_compt_min", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_min(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total)),
                ("p_compt_sum_nonfinal", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_sum(perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1])),
                ("p_compt_mean_nonfinal", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_mean(perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1])),
                ("p_compt_min_nonfinal", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_min(perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1])),
                ("-log_p_compt_sum", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_sum(-np.log(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total))),
                ("-log_p_compt_mean", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_mean(-np.log(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total))),
                ("-log_p_compt_max", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_max(-np.log(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total))),
                ("-log_p_compt_min", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_min(-np.log(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total))),
                ("-log_p_compt_sum_nonfinal", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_sum(-np.log(perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1]))),
                ("-log_p_compt_mean_nonfinal", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_mean(-np.log(perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1]))),
                ("-log_p_compt_min_nonfinal", ["linear_attenuation_compt", "lin_mu_total"], lambda: njit_min(-np.log(perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1]))),
            ],
        ),
        (
            12,
            [
                ("cross_total_sum", ["lin_mu_total"], lambda: njit_sum(perm_calc.lin_mu_total)),
                ("cross_total_mean", ["lin_mu_total"], lambda: njit_mean(perm_calc.lin_mu_total)),
                ("cross_total_max", ["lin_mu_total"], lambda: njit_max(perm_calc.lin_mu_total)),
                ("cross_total_ge_dist_sum", ["lin_mu_total", "ge_distance_perm"], lambda: njit_sum(perm_calc.lin_mu_total * perm_calc.ge_distance_perm)),
                ("cross_total_ge_dist_mean", ["lin_mu_total", "ge_distance_perm"], lambda: njit_mean(perm_calc.lin_mu_total * perm_calc.ge_distance_perm)),
                ("cross_total_ge_dist_max", ["lin_mu_total", "ge_distance_perm"], lambda: njit_max(perm_calc.lin_mu_total * perm_calc.ge_distance_perm)),
                ("cross_total_dist_sum", ["lin_mu_total", "distance_perm"], lambda: njit_sum(perm_calc.lin_mu_total * perm_calc.distance_perm)),
                ("cross_total_dist_mean", ["lin_mu_total", "distance_perm"], lambda: njit_mean(perm_calc.lin_mu_total * perm_calc.distance_perm)),
                ("cross_total_dist_max", ["lin_mu_total", "distance_perm"], lambda: njit_max(perm_calc.lin_mu_total * perm_calc.distance_perm)),
                ("cross_total_min", ["lin_mu_total"], lambda: njit_min(perm_calc.lin_mu_total)),
                ("cross_total_ge_dist_min", ["lin_mu_total", "ge_distance_perm"], lambda: njit_min(perm_calc.lin_mu_total * perm_calc.ge_distance_perm)),
                ("cross_total_dist_min", ["lin_mu_total", "distance_perm"], lambda: njit_min(perm_calc.lin_mu_total * perm_calc.distance_perm)),
            ],
        ),
        (
            32,
            [
                ("klein-nishina_rel_sum_sum", ["klein_nishina_relative_use_Ei"], lambda: njit_sum(perm_calc.klein_nishina_relative_use_Ei)),
                ("klein-nishina_rel_sum_mean", ["klein_nishina_relative_use_Ei"], lambda: njit_mean(perm_calc.klein_nishina_relative_use_Ei)),
                ("klein-nishina_rel_sum_max", ["klein_nishina_relative_use_Ei"], lambda: njit_max(perm_calc.klein_nishina_relative_use_Ei)),
                ("klein-nishina_rel_sum_min", ["klein_nishina_relative_use_Ei"], lambda: njit_min(perm_calc.klein_nishina_relative_use_Ei)),
                ("-log_klein-nishina_rel_sum_sum", ["klein_nishina_relative_use_Ei"], lambda: njit_sum(-np.log(perm_calc.klein_nishina_relative_use_Ei))),
                ("-log_klein-nishina_rel_sum_mean", ["klein_nishina_relative_use_Ei"], lambda: njit_mean(-np.log(perm_calc.klein_nishina_relative_use_Ei))),
                ("-log_klein-nishina_rel_sum_max", ["klein_nishina_relative_use_Ei"], lambda: njit_max(-np.log(perm_calc.klein_nishina_relative_use_Ei))),
                ("-log_klein-nishina_rel_sum_min", ["klein_nishina_relative_use_Ei"], lambda: njit_min(-np.log(perm_calc.klein_nishina_relative_use_Ei))),
                ("klein-nishina_rel_geo_sum", ["klein_nishina_relative"], lambda: njit_sum(perm_calc.klein_nishina_relative)),
                ("klein-nishina_rel_geo_mean", ["klein_nishina_relative"], lambda: njit_mean(perm_calc.klein_nishina_relative)),
                ("klein-nishina_rel_geo_max", ["klein_nishina_relative"], lambda: njit_max(perm_calc.klein_nishina_relative)),
                ("klein-nishina_rel_geo_min", ["klein_nishina_relative"], lambda: njit_min(perm_calc.klein_nishina_relative)),
                ("-log_klein-nishina_rel_geo_sum", ["klein_nishina_relative"], lambda: njit_sum(-np.log(perm_calc.klein_nishina_relative))),
                ("-log_klein-nishina_rel_geo_mean", ["klein_nishina_relative"], lambda: njit_mean(-np.log(perm_calc.klein_nishina_relative))),
                ("-log_klein-nishina_rel_geo_max", ["klein_nishina_relative"], lambda: njit_max(-np.log(perm_calc.klein_nishina_relative))),
                ("-log_klein-nishina_rel_geo_min", ["klein_nishina_relative"], lambda: njit_min(-np.log(perm_calc.klein_nishina_relative))),
                ("klein-nishina_sum_sum", ["klein_nishina_use_Ei"], lambda: njit_sum(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS)),
                ("klein-nishina_sum_mean", ["klein_nishina_use_Ei"], lambda: njit_mean(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS)),
                ("klein-nishina_sum_max", ["klein_nishina_use_Ei"], lambda: njit_max(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS)),
                ("klein-nishina_sum_min", ["klein_nishina_use_Ei"], lambda: njit_min(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS)),
                ("-log_klein-nishina_sum_sum", ["klein_nishina_use_Ei"], lambda: njit_sum(-np.log(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS))),
                ("-log_klein-nishina_sum_mean", ["klein_nishina_use_Ei"], lambda: njit_mean(-np.log(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS))),
                ("-log_klein-nishina_sum_max", ["klein_nishina_use_Ei"], lambda: njit_max(-np.log(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS))),
                ("-log_klein-nishina_sum_min", ["klein_nishina_use_Ei"], lambda: njit_min(-np.log(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS))),
                ("klein-nishina_geo_sum", ["klein_nishina"], lambda: njit_sum(perm_calc.klein_nishina * RANGE_PROCESS)),
                ("klein-nishina_geo_mean", ["klein_nishina"], lambda: njit_mean(perm_calc.klein_nishina * RANGE_PROCESS)),
                ("klein-nishina_geo_max", ["klein_nishina"], lambda: njit_max(perm_calc.klein_nishina * RANGE_PROCESS)),
                ("klein-nishina_geo_min", ["klein_nishina"], lambda: njit_min(perm_calc.klein_nishina * RANGE_PROCESS)),
                ("-log_klein-nishina_geo_sum", ["klein_nishina"], lambda: njit_sum(-np.log(perm_calc.klein_nishina * RANGE_PROCESS))),
                ("-log_klein-nishina_geo_mean", ["klein_nishina"], lambda: njit_mean(-np.log(perm_calc.klein_nishina * RANGE_PROCESS))),
                ("-log_klein-nishina_geo_max", ["klein_nishina"], lambda: njit_max(-np.log(perm_calc.klein_nishina * RANGE_PROCESS))),
                ("-log_klein-nishina_geo_min", ["klein_nishina"], lambda: njit_min(-np.log(perm_calc.klein_nishina * RANGE_PROCESS))),
            ],
        ),
        (
            12,
            [
                ("first_r", ["radii_perm"], lambda: perm_calc.radii_perm[0]),
                ("final_r", ["radii_perm"], lambda: perm_calc.radii_perm[-1]),
                ("first_energy_ratio", ["energies_perm", "energy_sum"], lambda: perm_calc.energies_perm[0] / perm_calc.energy_sum),
                ("final_energy_ratio", ["energies_perm"], lambda: perm_calc.energies_perm[-2] / (perm_calc.energies_perm[-2] + perm_calc.energies_perm[-1])),
                ("first_is_not_largest", ["energies_perm"], lambda: njit_any(perm_calc.energies_perm[1:] > perm_calc.energies_perm[0])),
                ("first_is_not_closest", ["radii_perm"], lambda: njit_any(perm_calc.radii_perm[1:] < perm_calc.radii_perm[0])),
                ("tango_variance", ["tango_estimates_perm"], lambda: np.var(perm_calc.tango_estimates_perm)),
                ("tango_v_variance", ["tango_estimates_sigma_perm"], lambda: 1.0 / njit_sum(1.0 / perm_calc.tango_estimates_sigma_perm**2)),
                ("tango_sigma", ["tango_estimates_perm"], lambda: np.std(perm_calc.tango_estimates_perm)),
                ("tango_v_sigma", ["tango_estimates_sigma_perm"], lambda: np.sqrt(1.0 / njit_sum(1.0 / perm_calc.tango_estimates_sigma_perm**2))),
                ("escape_probability", ["escape_probability"], lambda: perm_calc.escape_probability),
                ("-log_escape_probability", ["escape_probability"], lambda: -np.log(perm_calc.escape_probability + 1e-16)),
            ],
        ),
    ]

    for block_size, block in feature_groups:
        emit_feature_group([(block_size, features) for features in block])

    if name_mode:
        return names
    if dependency_mode:
        return dependencies_dict
    return features_vector
