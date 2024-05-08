"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Feature level computations
"""

from __future__ import annotations

from typing import Iterable, Optional

import numba  # TODO - are there good opportunities for JIT here?
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
    return njit_sum(stdev_weighted_value **2) / njit_sum(1 / stdev **2)


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

    Given permutation measurements, get values the values for features

    Args:
        - permutation: the permutation of indices
        - calc: the calculated permutation level values
        - Nmi: the number of interactions
        - boolean_vector: indicates which features to compute
        - name_mode: return the names of the features
        - dependency_mode: return a dictionary of computation dependencies
        - all_computations: perform all computations
        - number_of_values: used to create output feature vector

    Returns:
        - features_vector: values of computed features
        - names: names of features if name_mode
        - dependencies_dict: dictionary of computational dependencies if
          dependency_mode
    """

    compute_mode = not name_mode and not dependency_mode

    # if len(permutation) == 1:
    #     return None
    # if start_energy is None:
    #     start_energy = calc.energy_rev_cumsum[0]
    if compute_mode:
        if Nmi is None:
            Nmi = len(permutation)
        if len(permutation) == 1:
            return np.zeros((number_of_values,))

    if compute_mode and boolean_vector is None:
        all_computations = True

    features_vector = np.zeros((number_of_values,))
    if all_computations:
        boolean_vector = np.ones(features_vector.shape, dtype=np.bool_)

    names = []
    dependencies_dict = {}
    index = 0

    def compute_value(name, dependencies, compute_fn):
        """
        Computes the value of the
        """
        if compute_mode:
            if boolean_vector[index]:
                features_vector[index] = compute_fn()
        elif name_mode:
            names.append(name)
        elif dependency_mode:
            dependencies_dict[name] = dependencies

    # def compute_and_return_value(name, dependencies, compute_fn):
    #     """
    #     Computes the value of the
    #     """
    #     if compute_mode:
    #         if boolean_vector[index]:
    #             features_vector[index] = compute_fn()
    #             return features_vector[index]
    #         return 0.0
    #     elif name_mode:
    #         names.append(name)
    #     elif dependency_mode:
    #         dependencies_dict[name] = dependencies

    # 22 features
    # we can skip all of this if we aren't computing anything inside of here
    if compute_mode and not njit_any(boolean_vector[index : index + 22 + 12 + 12]):
        index += 22 + 12 + 12
    else:
        if compute_mode and not njit_any(boolean_vector[index : index + 22]):
            index += 22
        else:
            compute_value(
                "rsg_sum_1", ["res_sum_geo"], lambda: njit_sum(perm_calc.res_sum_geo)
            )
            index += 1

            compute_value(
                "rsg_sum_1_first", ["res_sum_geo"], lambda: perm_calc.res_sum_geo[0]
            )
            index += 1

            compute_value(
                "rsg_mean_1", ["res_sum_geo"], lambda: njit_mean(perm_calc.res_sum_geo)
            )
            index += 1

            compute_value(
                "rsg_mean_1_first",
                ["res_sum_geo"],
                lambda: perm_calc.res_sum_geo[0] / len(perm_calc.res_sum_geo),
            )
            index += 1

            compute_value(
                "rsg_wmean_1v",
                ["res_sum_geo_v", "res_sum_geo_sigma"],
                # lambda: njit_sum(perm_calc.res_sum_geo_v) / njit_sum(1.0 / perm_calc.res_sum_geo_sigma),
                lambda: wmean_1v_func(perm_calc.res_sum_geo_v, perm_calc.res_sum_geo_sigma),
            )
            index += 1

            # TODO - this is not right... currently equal to res_sum_geo[0]
            compute_value(
                "rsg_wmean_1v_first",
                ["res_sum_geo_v", "res_sum_geo_sigma"],
                lambda: perm_calc.res_sum_geo_v[0]
                / (1.0 / perm_calc.res_sum_geo_sigma[0]),
            )
            index += 1

            compute_value(
                "rsg_norm_2",
                ["res_sum_geo"],
                lambda: njit_norm(perm_calc.res_sum_geo) / Nmi,
            )
            index += 1

            compute_value(
                "rsg_sum_2", ["res_sum_geo"], lambda: njit_sum(perm_calc.res_sum_geo**2)
            )
            index += 1

            compute_value(
                "rsg_sum_2_first",
                ["res_sum_geo"],
                lambda: perm_calc.res_sum_geo[0] ** 2,
            )
            index += 1

            compute_value(
                "rsg_mean_2",
                ["res_sum_geo"],
                lambda: njit_mean(perm_calc.res_sum_geo**2),
            )
            index += 1

            compute_value(
                "rsg_mean_2_first",
                ["res_sum_geo"],
                lambda: perm_calc.res_sum_geo[0] ** 2 / len(perm_calc.res_sum_geo),
            )
            index += 1

            compute_value(
                "rsg_wmean_2v",
                ["res_sum_geo_v", "res_sum_geo_sigma"],
                # lambda: njit_sum(perm_calc.res_sum_geo_v**2) / njit_sum((1.0 / perm_calc.res_sum_geo_sigma) ** 2),
                lambda: wmean_2v_func(perm_calc.res_sum_geo_v, perm_calc.res_sum_geo_sigma),
            )
            index += 1

            compute_value(
                "rsg_wmean_2v_first",
                ["res_sum_geo_v", "res_sum_geo_sigma"],
                lambda: perm_calc.res_sum_geo_v[0] ** 2
                / ((1.0 / perm_calc.res_sum_geo_sigma[0]) ** 2),
            )
            index += 1

            compute_value(
                "rsg_sum_1v",
                ["res_sum_geo_v"],
                lambda: njit_sum(perm_calc.res_sum_geo_v),
            )
            index += 1

            compute_value(
                "rsg_sum_1v_first",
                ["res_sum_geo_v"],
                lambda: perm_calc.res_sum_geo_v[0],
            )
            index += 1

            compute_value(
                "rsg_mean_1v",
                ["res_sum_geo_v"],
                lambda: njit_mean(perm_calc.res_sum_geo_v),
            )
            index += 1

            compute_value(
                "rsg_mean_1v_first",
                ["res_sum_geo_v"],
                lambda: perm_calc.res_sum_geo_v[0] / len(perm_calc.res_sum_geo_v),
            )
            index += 1

            compute_value(
                "rsg_norm_2v",
                ["res_sum_geo_v"],
                lambda: njit_norm(perm_calc.res_sum_geo_v) / Nmi,
            )
            index += 1

            compute_value(
                "rsg_sum_2v",
                ["res_sum_geo_v"],
                lambda: njit_sum(perm_calc.res_sum_geo_v**2),
            )
            index += 1

            compute_value(
                "rsg_sum_2v_first",
                ["res_sum_geo_v"],
                lambda: perm_calc.res_sum_geo_v[0] ** 2,
            )
            index += 1

            compute_value(
                "rsg_mean_2v",
                ["res_sum_geo_v"],
                lambda: njit_mean(perm_calc.res_sum_geo_v**2),
            )
            index += 1

            compute_value(
                "rsg_mean_2v_first",
                ["res_sum_geo_v"],
                lambda: perm_calc.res_sum_geo_v[0] ** 2 / len(perm_calc.res_sum_geo_v),
            )
            index += 1

        # %%
        if compute_mode and not njit_any(boolean_vector[index : index + 12]):
            index += 12
        else:
            compute_value(
                "rsl_mean_1", ["res_sum_loc"], lambda: njit_mean(perm_calc.res_sum_loc)
            )
            index += 1

            compute_value(
                "rsl_sum_1", ["res_sum_loc"], lambda: njit_sum(perm_calc.res_sum_loc)
            )
            index += 1

            compute_value(
                "rsl_norm_2",
                ["res_sum_loc"],
                lambda: njit_norm(perm_calc.res_sum_loc) / Nmi,
            )
            index += 1

            compute_value(
                "rsl_sum_2", ["res_sum_loc"], lambda: njit_sum(perm_calc.res_sum_loc**2)
            )
            index += 1

            compute_value(
                "rsl_mean_2",
                ["res_sum_loc"],
                lambda: njit_mean(perm_calc.res_sum_loc**2),
            )
            index += 1

            compute_value(
                "rsl_sum_1v",
                ["res_sum_loc_v"],
                lambda: njit_sum(perm_calc.res_sum_loc_v),
            )
            index += 1

            compute_value(
                "rsl_mean_1v",
                ["res_sum_loc_v"],
                lambda: njit_mean(perm_calc.res_sum_loc_v),
            )
            index += 1

            compute_value(
                "rsl_norm_2v",
                ["res_sum_loc_v"],
                lambda: njit_norm(perm_calc.res_sum_loc_v)
                / Nmi
                / np.sqrt(len(perm_calc.res_sum_loc_v)),
            )
            index += 1

            compute_value(
                "rsl_mean_2v",
                ["res_sum_loc_v"],
                lambda: njit_mean(perm_calc.res_sum_loc_v**2),
            )
            index += 1

            compute_value(
                "rsl_sum_2v",
                ["res_sum_loc_v"],
                lambda: njit_sum(perm_calc.res_sum_loc_v**2),
            )
            index += 1

            compute_value(
                "rsl_wmean_2v",
                ["res_sum_loc_v", "res_sum_loc_sigma"],
                # lambda: njit_sum(perm_calc.res_sum_loc_v**2) / njit_sum((1.0 / perm_calc.res_sum_loc_sigma) ** 2),
                lambda: wmean_2v_func(perm_calc.res_sum_loc_v, perm_calc.res_sum_loc_sigma),
            )
            index += 1

            compute_value(
                "rsl_wmean_1v",
                ["res_sum_loc_v", "res_sum_loc_sigma"],
                # lambda: njit_sum(perm_calc.res_sum_loc_v) / njit_sum(1.0 / perm_calc.res_sum_loc_sigma),
                lambda: wmean_1v_func(perm_calc.res_sum_loc_v, perm_calc.res_sum_loc_sigma),
            )
            index += 1

        # %%
        if compute_mode and not njit_any(boolean_vector[index : index + 12]):
            index += 12
        else:
            compute_value(
                "rlg_sum_1v",
                ["res_loc_geo_v"],
                lambda: njit_sum(perm_calc.res_loc_geo_v),
            )
            index += 1

            compute_value(
                "rlg_mean_1v",
                ["res_loc_geo_v"],
                lambda: njit_mean(perm_calc.res_loc_geo_v),
            )
            index += 1

            compute_value(
                "rlg_norm_2v",
                ["res_loc_geo_v"],
                lambda: njit_norm(perm_calc.res_loc_geo_v)
                / Nmi
                / np.sqrt(len(perm_calc.res_loc_geo_v)),
            )
            index += 1

            compute_value(
                "rlg_sum_2v",
                ["res_loc_geo_v"],
                lambda: njit_sum(perm_calc.res_loc_geo_v**2),
            )
            index += 1

            compute_value(
                "rlg_mean_2v",
                ["res_loc_geo_v"],
                lambda: njit_mean(perm_calc.res_loc_geo_v**2),
            )
            index += 1

            compute_value(
                "rlg_sum_1", ["res_loc_geo"], lambda: njit_sum(perm_calc.res_loc_geo)
            )
            index += 1

            compute_value(
                "rlg_mean_1", ["res_loc_geo"], lambda: njit_mean(perm_calc.res_loc_geo)
            )
            index += 1

            compute_value(
                "rlg_norm_2",
                ["res_loc_geo"],
                lambda: njit_norm(perm_calc.res_loc_geo) / Nmi,
            )
            index += 1

            compute_value(
                "rlg_wmean_1v",
                ["res_loc_geo_v", "res_loc_geo_sigma"],
                # lambda: njit_sum(perm_calc.res_loc_geo_v) / njit_sum(1.0 / perm_calc.res_loc_geo_sigma),
                lambda: wmean_1v_func(perm_calc.res_loc_geo_v, perm_calc.res_loc_geo_sigma),
            )
            index += 1

            compute_value(
                "rlg_sum_2", ["res_loc_geo"], lambda: njit_sum(perm_calc.res_loc_geo**2)
            )
            index += 1

            compute_value(
                "rlg_mean_2",
                ["res_loc_geo"],
                lambda: njit_mean(perm_calc.res_loc_geo**2),
            )
            index += 1

            compute_value(
                "rlg_wmean_2v",
                ["res_loc_geo_v", "res_loc_geo_sigma"],
                # lambda: njit_sum(perm_calc.res_loc_geo_v**2) / njit_sum((1.0 / perm_calc.res_loc_geo_sigma) ** 2),
                lambda: wmean_2v_func(perm_calc.res_loc_geo_v, perm_calc.res_loc_geo_sigma),
            )
            index += 1

    # %% Compton penalty
    if compute_mode and not njit_any(
        boolean_vector[index : index + 6 + 22 + 12 + 22 + 12]
    ):
        index += 6 + 22 + 12 + 22 + 12
    else:
        if compute_mode and not njit_any(boolean_vector[index : index + 6]):
            index += 6
        else:
            compute_value(
                "c_penalty_sum_1",
                ["compton_penalty"],
                lambda: njit_sum(perm_calc.compton_penalty),
            )
            index += 1

            compute_value(
                "c_penalty_mean_1",
                ["compton_penalty"],
                lambda: njit_mean(perm_calc.compton_penalty),
            )
            index += 1

            compute_value(
                "c_penalty_ell_sum_1",
                ["compton_penalty_ell1"],
                lambda: njit_sum(perm_calc.compton_penalty_ell1),
            )
            index += 1

            compute_value(
                "c_penalty_ell_mean_1",
                ["compton_penalty_ell1"],
                lambda: njit_mean(perm_calc.compton_penalty_ell1),
            )
            index += 1

            compute_value(
                "c_penalty_ell_sum_2",
                ["compton_penalty_ell1"],
                lambda: njit_sum(perm_calc.compton_penalty_ell1**2),
            )
            index += 1

            compute_value(
                "c_penalty_ell_mean_2",
                ["compton_penalty_ell1"],
                lambda: njit_mean(perm_calc.compton_penalty_ell1**2),
            )
            index += 1

        # %%
        if compute_mode and not njit_any(boolean_vector[index : index + 22]):
            index += 22
        else:
            compute_value("rc_sum_1", ["res_cos"], lambda: njit_sum(perm_calc.res_cos))
            index += 1

            compute_value(
                "rc_mean_1", ["res_cos"], lambda: njit_mean(perm_calc.res_cos)
            )
            index += 1

            compute_value(
                "rc_norm_2", ["res_cos"], lambda: njit_norm(perm_calc.res_cos) / Nmi
            )
            index += 1

            compute_value(
                "rc_sum_2", ["res_cos"], lambda: njit_sum(perm_calc.res_cos**2)
            )
            index += 1

            compute_value(
                "rc_mean_2", ["res_cos"], lambda: njit_mean(perm_calc.res_cos**2)
            )
            index += 1

            compute_value(
                "rc_sum_1_penalty_removed",
                ["res_cos", "compton_penalty"],
                lambda: njit_sum(perm_calc.res_cos * (1.0 - perm_calc.compton_penalty)),
            )
            index += 1

            compute_value(
                "rc_mean_1_penalty_removed",
                ["res_cos", "compton_penalty"],
                lambda: njit_mean(
                    perm_calc.res_cos * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rc_sum_2_penalty_removed",
                ["res_cos", "compton_penalty"],
                lambda: njit_sum(
                    perm_calc.res_cos**2 * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rc_mean_2_penalty_removed",
                ["res_cos", "compton_penalty"],
                lambda: njit_mean(
                    perm_calc.res_cos**2 * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rc_wmean_1v",
                ["res_cos_v", "res_cos_sigma"],
                # lambda: njit_sum(perm_calc.res_cos_v) / njit_sum(1.0 / perm_calc.res_cos_sigma),
                lambda: wmean_1v_func(perm_calc.res_cos_v, perm_calc.res_cos_sigma),
            )
            index += 1

            compute_value(
                "rc_wmean_2v",
                ["res_cos_v", "res_cos_sigma"],
                # lambda: njit_sum(perm_calc.res_cos_v**2) / njit_sum((1.0 / perm_calc.res_cos_sigma) ** 2),
                lambda: wmean_2v_func(perm_calc.res_cos_v, perm_calc.res_cos_sigma),
            )
            index += 1

            compute_value(
                "rc_sum_1v", ["res_cos_v"], lambda: njit_sum(perm_calc.res_cos_v)
            )
            index += 1

            compute_value(
                "rc_mean_1v", ["res_cos_v"], lambda: njit_mean(perm_calc.res_cos_v)
            )
            index += 1

            compute_value(
                "rc_norm_2v",
                ["res_cos_v"],
                lambda: njit_norm(perm_calc.res_cos_v) / Nmi,
            )
            index += 1

            compute_value(
                "rc_sum_2v", ["res_cos_v"], lambda: njit_sum(perm_calc.res_cos_v**2)
            )
            index += 1

            compute_value(
                "rc_mean_2v", ["res_cos_v"], lambda: njit_mean(perm_calc.res_cos_v**2)
            )
            index += 1

            compute_value(
                "rc_wmean_1v_penalty_removed",
                ["compton_penalty", "res_cos_v", "res_cos_sigma"],
                lambda: rc_wmean_1v_penalty_removed_func(
                    perm_calc.compton_penalty,
                    perm_calc.res_cos_v,
                    perm_calc.res_cos_sigma,
                ),
            )
            index += 1

            compute_value(
                "rc_wmean_2v_penalty_removed",
                ["compton_penalty", "res_cos_v", "res_cos_sigma"],
                lambda: rc_wmean_2v_penalty_removed_func(
                    perm_calc.compton_penalty,
                    perm_calc.res_cos_v,
                    perm_calc.res_cos_sigma,
                ),
            )
            index += 1

            compute_value(
                "rc_sum_1v_penalty_removed",
                ["res_cos_v", "res_cos_sigma"],
                lambda: njit_sum(
                    perm_calc.res_cos_v * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rc_mean_1v_penalty_removed",
                ["res_cos_v", "res_cos_sigma"],
                lambda: njit_mean(
                    perm_calc.res_cos_v * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rc_sum_2v_penalty_removed",
                ["res_cos_v", "res_cos_sigma"],
                lambda: njit_sum(
                    perm_calc.res_cos_v**2 * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rc_mean_2v_penalty_removed",
                ["res_cos_v", "res_cos_sigma"],
                lambda: njit_mean(
                    perm_calc.res_cos_v**2 * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

        # %%

        if compute_mode and not njit_any(boolean_vector[index : index + 12]):
            index += 12
        else:
            compute_value(
                "rc_cap_sum_1", ["res_cos_cap"], lambda: njit_sum(perm_calc.res_cos_cap)
            )
            index += 1

            compute_value(
                "rc_cap_mean_1",
                ["res_cos_cap"],
                lambda: njit_mean(perm_calc.res_cos_cap),
            )
            index += 1

            compute_value(
                "rc_cap_norm_2",
                ["res_cos_cap"],
                lambda: njit_norm(perm_calc.res_cos_cap) / Nmi,
            )
            index += 1

            compute_value(
                "rc_cap_sum_2",
                ["res_cos_cap"],
                lambda: njit_sum(perm_calc.res_cos_cap**2),
            )
            index += 1

            compute_value(
                "rc_cap_mean_2",
                ["res_cos_cap"],
                lambda: njit_mean(perm_calc.res_cos_cap**2),
            )
            index += 1

            compute_value(
                "rc_cap_wmean_1v",
                ["res_cos_cap_v", "res_cos_sigma"],
                # lambda: njit_sum(perm_calc.res_cos_cap_v) / njit_sum(1.0 / perm_calc.res_cos_sigma),
                lambda: wmean_1v_func(perm_calc.res_cos_cap_v, perm_calc.res_cos_sigma),
            )
            index += 1

            compute_value(
                "rc_cap_wmean_2v",
                ["res_cos_cap_v", "res_cos_sigma"],
                # lambda: njit_sum(perm_calc.res_cos_cap_v**2) / njit_sum((1.0 / perm_calc.res_cos_sigma) ** 2),
                lambda: wmean_2v_func(perm_calc.res_cos_cap_v, perm_calc.res_cos_sigma),
            )
            index += 1

            compute_value(
                "rc_cap_sum_1v",
                ["res_cos_cap_v"],
                lambda: njit_sum(perm_calc.res_cos_cap_v),
            )
            index += 1

            compute_value(
                "rc_cap_mean_1v",
                ["res_cos_cap_v"],
                lambda: njit_mean(perm_calc.res_cos_cap_v),
            )
            index += 1

            compute_value(
                "rc_cap_norm_2v",
                ["res_cos_cap_v"],
                lambda: njit_norm(perm_calc.res_cos_cap_v) / Nmi,
            )
            index += 1

            compute_value(
                "rc_cap_sum_2v",
                ["res_cos_cap_v"],
                lambda: njit_sum(perm_calc.res_cos_cap_v**2),
            )
            index += 1

            compute_value(
                "rc_cap_mean_2v",
                ["res_cos_cap_v"],
                lambda: njit_mean(perm_calc.res_cos_cap_v**2),
            )
            index += 1

        # %%

        if compute_mode and not njit_any(boolean_vector[index : index + 22]):
            index += 22
        else:
            compute_value(
                "rth_sum_1", ["res_theta"], lambda: njit_sum(perm_calc.res_theta)
            )
            index += 1

            compute_value(
                "rth_mean_1", ["res_theta"], lambda: njit_mean(perm_calc.res_theta)
            )
            index += 1

            compute_value(
                "rth_norm_2",
                ["res_theta"],
                lambda: njit_norm(perm_calc.res_theta) / Nmi,
            )
            index += 1

            compute_value(
                "rth_sum_2", ["res_theta"], lambda: njit_sum(perm_calc.res_theta**2)
            )
            index += 1

            compute_value(
                "rth_mean_2", ["res_theta"], lambda: njit_mean(perm_calc.res_theta**2)
            )
            index += 1

            compute_value(
                "rth_sum_1_penalty_removed",
                ["res_theta", "compton_penalty"],
                lambda: njit_sum(
                    perm_calc.res_theta * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rth_mean_1_penalty_removed",
                ["res_theta", "compton_penalty"],
                lambda: njit_mean(
                    perm_calc.res_theta * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rth_sum_2_penalty_removed",
                ["res_theta", "compton_penalty"],
                lambda: njit_sum(
                    perm_calc.res_theta**2 * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rth_mean_2_penalty_removed",
                ["res_theta", "compton_penalty"],
                lambda: njit_mean(
                    perm_calc.res_theta**2 * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rth_wmean_1v",
                ["res_theta_v", "res_theta_sigma"],
                # lambda: njit_sum(perm_calc.res_theta_v) / njit_sum(1.0 / perm_calc.res_theta_sigma),
                lambda: wmean_1v_func(perm_calc.res_theta_v, perm_calc.res_theta_sigma),
            )
            index += 1

            compute_value(
                "rth_wmean_2v",
                ["res_theta_v", "res_theta_sigma"],
                # lambda: njit_sum(perm_calc.res_theta_v**2) / njit_sum((1.0 / perm_calc.res_theta_sigma) ** 2),
                lambda: wmean_2v_func(perm_calc.res_theta_v, perm_calc.res_theta_sigma),
            )
            index += 1

            compute_value(
                "rth_sum_1v", ["res_theta_v"], lambda: njit_sum(perm_calc.res_theta_v)
            )
            index += 1

            compute_value(
                "rth_mean_1v", ["res_theta_v"], lambda: njit_mean(perm_calc.res_theta_v)
            )
            index += 1

            compute_value(
                "rth_norm_2v",
                ["res_theta_v"],
                lambda: njit_norm(perm_calc.res_theta_v) / Nmi,
            )
            index += 1

            compute_value(
                "rth_sum_2v",
                ["res_theta_v"],
                lambda: njit_sum(perm_calc.res_theta_v**2),
            )
            index += 1

            compute_value(
                "rth_mean_2v",
                ["res_theta_v"],
                lambda: njit_mean(perm_calc.res_theta_v**2),
            )
            index += 1

            compute_value(
                "rth_sum_1v_penalty_removed",
                ["res_theta_v", "compton_penalty"],
                lambda: njit_sum(
                    perm_calc.res_theta_v * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rth_mean_1v_penalty_removed",
                ["res_theta_v", "compton_penalty"],
                lambda: njit_mean(
                    perm_calc.res_theta_v * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rth_sum_2v_penalty_removed",
                ["res_theta_v", "compton_penalty"],
                lambda: njit_sum(
                    perm_calc.res_theta_v**2 * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rth_mean_2v_penalty_removed",
                ["res_theta_v", "compton_penalty"],
                lambda: njit_mean(
                    perm_calc.res_theta_v**2 * (1.0 - perm_calc.compton_penalty)
                ),
            )
            index += 1

            compute_value(
                "rth_wmean_1v_penalty_removed",
                ["compton_penalty", "res_theta_v", "res_theta_sigma"],
                lambda: rth_wmean_1v_penalty_removed_func(
                    perm_calc.compton_penalty,
                    perm_calc.res_theta_v,
                    perm_calc.res_theta_sigma,
                ),
            )
            index += 1

            compute_value(
                "rth_wmean_2v_penalty_removed",
                ["compton_penalty", "res_theta_v", "res_theta_sigma"],
                lambda: rth_wmean_2v_penalty_removed_func(
                    perm_calc.compton_penalty,
                    perm_calc.res_theta_v,
                    perm_calc.res_theta_sigma,
                ),
            )
            index += 1

        # %%

        if compute_mode and not njit_any(boolean_vector[index : index + 12]):
            index += 12
        else:
            compute_value(
                "rth_cap_sum_1",
                ["res_theta_cap"],
                lambda: njit_sum(perm_calc.res_theta_cap),
            )
            index += 1

            compute_value(
                "rth_cap_mean_1",
                ["res_theta_cap"],
                lambda: njit_mean(perm_calc.res_theta_cap),
            )
            index += 1

            compute_value(
                "rth_cap_norm_2",
                ["res_theta_cap"],
                lambda: njit_norm(perm_calc.res_theta_cap) / Nmi,
            )
            index += 1

            compute_value(
                "rth_cap_sum_2",
                ["res_theta_cap"],
                lambda: njit_sum(perm_calc.res_theta_cap**2),
            )
            index += 1

            compute_value(
                "rth_cap_mean_2",
                ["res_theta_cap"],
                lambda: njit_mean(perm_calc.res_theta_cap**2),
            )
            index += 1

            compute_value(
                "rth_cap_wmean_1v",
                ["res_theta_cap_v", "res_theta_sigma"],
                lambda: wmean_1v_func(
                    perm_calc.res_theta_cap_v, perm_calc.res_theta_sigma
                ),
            )
            index += 1

            compute_value(
                "rth_cap_wmean_2v",
                ["res_theta_cap_v", "res_theta_sigma"],
                lambda: wmean_2v_func(
                    perm_calc.res_theta_cap_v, perm_calc.res_theta_sigma
                ),
            )
            index += 1

            compute_value(
                "rth_cap_sum_1v",
                ["res_theta_cap_v"],
                lambda: njit_sum(perm_calc.res_theta_cap_v),
            )
            index += 1

            compute_value(
                "rth_cap_mean_1v",
                ["res_theta_cap_v"],
                lambda: njit_mean(perm_calc.res_theta_cap_v),
            )
            index += 1

            compute_value(
                "rth_cap_norm_2v",
                ["res_theta_cap_v"],
                lambda: njit_norm(perm_calc.res_theta_cap_v) / Nmi,
            )
            index += 1

            compute_value(
                "rth_cap_sum_2v",
                ["res_theta_cap_v"],
                lambda: njit_sum(perm_calc.res_theta_cap_v**2),
            )
            index += 1

            compute_value(
                "rth_cap_mean_2v",
                ["res_theta_cap_v"],
                lambda: njit_mean(perm_calc.res_theta_cap_v**2),
            )
            index += 1

    if compute_mode and not njit_any(boolean_vector[index : index + 4 + 15 + 21]):
        index += 4 + 15 + 21
    else:
        # %% Distances (Euclidean and Germanium)

        if compute_mode and not njit_any(boolean_vector[index : index + 4]):
            index += 4
        else:
            compute_value(
                "distances_sum",
                ["distance_perm"],
                lambda: njit_sum(perm_calc.distance_perm),
            )
            index += 1

            compute_value(
                "distances_mean",
                ["distance_perm"],
                lambda: njit_mean(perm_calc.distance_perm),
            )
            index += 1

            compute_value(
                "ge_distances_sum",
                ["ge_distance_perm"],
                lambda: njit_sum(perm_calc.ge_distance_perm),
            )
            index += 1

            compute_value(
                "ge_distances_mean",
                ["ge_distance_perm"],
                lambda: njit_mean(perm_calc.ge_distance_perm),
            )
            index += 1

        # %% Attenuation coefficients and cross-sections

        if compute_mode and not njit_any(boolean_vector[index : index + 15]):
            index += 15
        else:
            compute_value(
                "cross_abs_sum",
                ["linear_attenuation_abs"],
                lambda: njit_sum(perm_calc.linear_attenuation_abs),
            )
            index += 1

            compute_value(
                "cross_abs_final",
                ["linear_attenuation_abs"],
                lambda: perm_calc.linear_attenuation_abs[-1],
            )
            index += 1

            compute_value(
                "cross_abs_mean",
                ["linear_attenuation_abs"],
                lambda: njit_mean(perm_calc.linear_attenuation_abs),
            )
            index += 1

            compute_value(
                "cross_abs_max",
                ["linear_attenuation_abs"],
                lambda: njit_max(perm_calc.linear_attenuation_abs),
            )
            index += 1

            compute_value(
                "cross_abs_ge_dist_sum",
                ["linear_attenuation_abs", "ge_distance_perm"],
                lambda: njit_sum(
                    perm_calc.linear_attenuation_abs * perm_calc.ge_distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_abs_ge_dist_final",
                ["linear_attenuation_abs", "ge_distance_perm"],
                lambda: perm_calc.linear_attenuation_abs[-1]
                * perm_calc.ge_distance_perm[-1],
            )
            index += 1

            compute_value(
                "cross_abs_ge_dist_mean",
                ["linear_attenuation_abs", "ge_distance_perm"],
                lambda: njit_mean(
                    perm_calc.linear_attenuation_abs * perm_calc.ge_distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_abs_ge_dist_max",
                ["linear_attenuation_abs", "ge_distance_perm"],
                lambda: njit_max(
                    perm_calc.linear_attenuation_abs * perm_calc.ge_distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_abs_dist_sum",
                ["linear_attenuation_abs", "distance_perm"],
                lambda: njit_sum(
                    perm_calc.linear_attenuation_abs * perm_calc.distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_abs_dist_final",
                ["linear_attenuation_abs", "distance_perm"],
                lambda: perm_calc.linear_attenuation_abs[-1]
                * perm_calc.distance_perm[-1],
            )
            index += 1

            compute_value(
                "cross_abs_dist_mean",
                ["linear_attenuation_abs", "distance_perm"],
                lambda: njit_mean(
                    perm_calc.linear_attenuation_abs * perm_calc.distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_abs_dist_max",
                ["linear_attenuation_abs", "distance_perm"],
                lambda: njit_max(
                    perm_calc.linear_attenuation_abs * perm_calc.distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_abs_min",
                ["linear_attenuation_abs"],
                lambda: njit_min(perm_calc.linear_attenuation_abs),
            )
            index += 1

            compute_value(
                "cross_abs_ge_dist_min",
                ["linear_attenuation_abs", "ge_distance_perm"],
                lambda: njit_min(
                    perm_calc.linear_attenuation_abs * perm_calc.ge_distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_abs_dist_min",
                ["linear_attenuation_abs", "distance_perm"],
                lambda: njit_min(
                    perm_calc.linear_attenuation_abs * perm_calc.distance_perm
                ),
            )
            index += 1

        # %% Compton

        if compute_mode and not njit_any(boolean_vector[index : index + 21]):
            index += 21
        else:
            compute_value(
                "cross_compt_sum",
                ["linear_attenuation_compt"],
                lambda: njit_sum(perm_calc.linear_attenuation_compt),
            )
            index += 1

            compute_value(
                "cross_compt_mean",
                ["linear_attenuation_compt"],
                lambda: njit_mean(perm_calc.linear_attenuation_compt),
            )
            index += 1

            compute_value(
                "cross_compt_max",
                ["linear_attenuation_compt"],
                lambda: njit_max(perm_calc.linear_attenuation_compt),
            )
            index += 1

            compute_value(
                "cross_compt_ge_dist_sum",
                ["linear_attenuation_compt", "ge_distance_perm"],
                lambda: njit_sum(
                    perm_calc.linear_attenuation_compt * perm_calc.ge_distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_compt_ge_dist_mean",
                ["linear_attenuation_compt", "ge_distance_perm"],
                lambda: njit_mean(
                    perm_calc.linear_attenuation_compt * perm_calc.ge_distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_compt_ge_dist_max",
                ["linear_attenuation_compt", "ge_distance_perm"],
                lambda: njit_max(
                    perm_calc.linear_attenuation_compt * perm_calc.ge_distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_compt_dist_sum",
                ["linear_attenuation_compt", "distance_perm"],
                lambda: njit_sum(
                    perm_calc.linear_attenuation_compt * perm_calc.distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_compt_dist_mean",
                ["linear_attenuation_compt", "distance_perm"],
                lambda: njit_mean(
                    perm_calc.linear_attenuation_compt * perm_calc.distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_compt_dist_max",
                ["linear_attenuation_compt", "distance_perm"],
                lambda: njit_max(
                    perm_calc.linear_attenuation_compt * perm_calc.distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_compt_min",
                ["linear_attenuation_compt"],
                lambda: njit_min(perm_calc.linear_attenuation_compt),
            )
            index += 1

            compute_value(
                "cross_compt_ge_dist_min",
                ["linear_attenuation_compt", "ge_distance_perm"],
                lambda: njit_min(
                    perm_calc.linear_attenuation_compt * perm_calc.ge_distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_compt_dist_min",
                ["linear_attenuation_compt", "distance_perm"],
                lambda: njit_min(
                    perm_calc.linear_attenuation_compt * perm_calc.distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_compt_sum_nonfinal",
                ["linear_attenuation_compt"],
                lambda: njit_sum(perm_calc.linear_attenuation_compt[:-1]),
            )
            index += 1

            compute_value(
                "cross_compt_mean_nonfinal",
                ["linear_attenuation_compt"],
                lambda: njit_mean(perm_calc.linear_attenuation_compt[:-1]),
            )
            index += 1

            compute_value(
                "cross_compt_min_nonfinal",
                ["linear_attenuation_compt"],
                lambda: njit_min(perm_calc.linear_attenuation_compt[:-1]),
            )
            index += 1

            compute_value(
                "cross_compt_dist_sum_nonfinal",
                ["linear_attenuation_compt", "distance_perm"],
                lambda: njit_sum(
                    perm_calc.linear_attenuation_compt[:-1]
                    * perm_calc.distance_perm[:-1]
                ),
            )
            index += 1

            compute_value(
                "cross_compt_dist_mean_nonfinal",
                ["linear_attenuation_compt", "distance_perm"],
                lambda: njit_mean(
                    perm_calc.linear_attenuation_compt[:-1]
                    * perm_calc.distance_perm[:-1]
                ),
            )
            index += 1

            compute_value(
                "cross_compt_dist_min_nonfinal",
                ["linear_attenuation_compt", "distance_perm"],
                lambda: njit_min(
                    perm_calc.linear_attenuation_compt[:-1]
                    * perm_calc.distance_perm[:-1]
                ),
            )
            index += 1

            compute_value(
                "cross_compt_ge_dist_sum_nonfinal",
                ["linear_attenuation_compt", "ge_distance_perm"],
                lambda: njit_sum(
                    perm_calc.linear_attenuation_compt[:-1]
                    * perm_calc.ge_distance_perm[:-1]
                ),
            )
            index += 1

            compute_value(
                "cross_compt_ge_dist_mean_nonfinal",
                ["linear_attenuation_compt", "ge_distance_perm"],
                lambda: njit_mean(
                    perm_calc.linear_attenuation_compt[:-1]
                    * perm_calc.ge_distance_perm[:-1]
                ),
            )
            index += 1

            compute_value(
                "cross_compt_ge_dist_min_nonfinal",
                ["linear_attenuation_compt", "ge_distance_perm"],
                lambda: njit_min(
                    perm_calc.linear_attenuation_compt[:-1]
                    * perm_calc.ge_distance_perm[:-1]
                ),
            )
            index += 1

    # %% pair production
    if False:  # pair production is not always useful to include
        if compute_mode and not njit_any(boolean_vector[index : index + 11]):
            index += 11
        else:
            compute_value(
                "cross_pair_sum",
                ["linear_attenuation_pair"],
                lambda: njit_sum(perm_calc.linear_attenuation_pair),
            )
            index += 1

            compute_value(
                "cross_pair_mean",
                ["linear_attenuation_pair"],
                lambda: njit_mean(perm_calc.linear_attenuation_pair),
            )
            index += 1

            compute_value(
                "cross_pair_max",
                ["linear_attenuation_pair"],
                lambda: njit_max(perm_calc.linear_attenuation_pair),
            )
            index += 1

            compute_value(
                "cross_pair_min",
                ["linear_attenuation_pair"],
                lambda: njit_min(perm_calc.linear_attenuation_pair),
            )
            index += 1

            compute_value(
                "cross_pair_dist_sum",
                ["linear_attenuation_pair", "distance_perm"],
                lambda: njit_sum(
                    perm_calc.linear_attenuation_pair * perm_calc.distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_pair_dist_mean",
                ["linear_attenuation_pair", "distance_perm"],
                lambda: njit_mean(
                    perm_calc.linear_attenuation_pair * perm_calc.distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_pair_dist_max",
                ["linear_attenuation_pair", "distance_perm"],
                lambda: njit_max(
                    perm_calc.linear_attenuation_pair * perm_calc.distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_pair_dist_min",
                ["linear_attenuation_pair", "distance_perm"],
                lambda: njit_min(
                    perm_calc.linear_attenuation_pair * perm_calc.distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_pair_ge_dist_sum",
                ["linear_attenuation_pair", "ge_distance_perm"],
                lambda: njit_sum(
                    perm_calc.linear_attenuation_pair * perm_calc.ge_distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_pair_ge_dist_mean",
                ["linear_attenuation_pair", "ge_distance_perm"],
                lambda: njit_mean(
                    perm_calc.linear_attenuation_pair * perm_calc.ge_distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_pair_ge_dist_max",
                ["linear_attenuation_pair", "ge_distance_perm"],
                lambda: njit_max(
                    perm_calc.linear_attenuation_pair * perm_calc.ge_distance_perm
                ),
            )
            index += 1

            compute_value(
                "cross_pair_ge_dist_min",
                ["linear_attenuation_pair", "ge_distance_perm"],
                lambda: njit_min(
                    perm_calc.linear_attenuation_pair * perm_calc.ge_distance_perm
                ),
            )
            index += 1

    if compute_mode and not njit_any(boolean_vector[index : index + 24]):
        index += 24
    else:
        compute_value(
            "p_abs_sum",
            ["linear_attenuation_abs", "lin_mu_total"],
            lambda: njit_sum(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total),
        )
        index += 1

        compute_value(
            "p_abs_final",
            ["linear_attenuation_abs", "lin_mu_total"],
            lambda: perm_calc.linear_attenuation_abs[-1] / perm_calc.lin_mu_total[-1],
        )
        index += 1

        compute_value(
            "p_abs_mean",
            ["linear_attenuation_abs", "lin_mu_total"],
            lambda: njit_mean(
                perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total
            ),
        )
        index += 1

        compute_value(
            "p_abs_max",
            ["linear_attenuation_abs", "lin_mu_total"],
            lambda: njit_max(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total),
        )
        index += 1

        compute_value(
            "p_abs_min",
            ["linear_attenuation_abs", "lin_mu_total"],
            lambda: njit_min(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total),
        )
        index += 1

        compute_value(
            "-log_p_abs_sum",
            ["linear_attenuation_abs", "lin_mu_total"],
            lambda: njit_sum(
                -np.log(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total)
            ),
        )
        index += 1

        compute_value(
            "-log_p_abs_final",
            ["linear_attenuation_abs", "lin_mu_total"],
            lambda: -np.log(
                perm_calc.linear_attenuation_abs[-1] / perm_calc.lin_mu_total[-1]
            ),
        )
        index += 1

        compute_value(
            "-log_p_abs_mean",
            ["linear_attenuation_abs", "lin_mu_total"],
            lambda: njit_mean(
                -np.log(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total)
            ),
        )
        index += 1

        compute_value(
            "-log_p_abs_max",
            ["linear_attenuation_abs", "lin_mu_total"],
            lambda: njit_max(
                -np.log(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total)
            ),
        )
        index += 1

        compute_value(
            "-log_p_abs_min",
            ["linear_attenuation_abs", "lin_mu_total"],
            lambda: njit_min(
                -np.log(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total)
            ),
        )
        index += 1

        compute_value(
            "p_compt_sum",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_sum(
                perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total
            ),
        )
        index += 1

        compute_value(
            "p_compt_mean",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_mean(
                perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total
            ),
        )
        index += 1

        compute_value(
            "p_compt_max",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_max(
                perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total
            ),
        )
        index += 1

        compute_value(
            "p_compt_min",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_min(
                perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total
            ),
        )
        index += 1

        compute_value(
            "p_compt_sum_nonfinal",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_sum(
                perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1]
            ),
        )
        index += 1

        compute_value(
            "p_compt_mean_nonfinal",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_mean(
                perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1]
            ),
        )
        index += 1

        compute_value(
            "p_compt_min_nonfinal",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_min(
                perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1]
            ),
        )
        index += 1

        compute_value(
            "-log_p_compt_sum",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_sum(
                -np.log(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total)
            ),
        )
        index += 1

        compute_value(
            "-log_p_compt_mean",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_mean(
                -np.log(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total)
            ),
        )
        index += 1

        compute_value(
            "-log_p_compt_max",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_max(
                -np.log(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total)
            ),
        )
        index += 1

        compute_value(
            "-log_p_compt_min",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_min(
                -np.log(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total)
            ),
        )
        index += 1

        compute_value(
            "-log_p_compt_sum_nonfinal",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_sum(
                -np.log(
                    perm_calc.linear_attenuation_compt[:-1]
                    / perm_calc.lin_mu_total[:-1]
                )
            ),
        )
        index += 1

        compute_value(
            "-log_p_compt_mean_nonfinal",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_mean(
                -np.log(
                    perm_calc.linear_attenuation_compt[:-1]
                    / perm_calc.lin_mu_total[:-1]
                )
            ),
        )
        index += 1

        compute_value(
            "-log_p_compt_min_nonfinal",
            ["linear_attenuation_compt", "lin_mu_total"],
            lambda: njit_min(
                -np.log(
                    perm_calc.linear_attenuation_compt[:-1]
                    / perm_calc.lin_mu_total[:-1]
                )
            ),
        )
        index += 1

    if False:
        if compute_mode and not njit_any(boolean_vector[index : index + 8]):
            index += 8
        else:
            compute_value(
                "p_pair_sum",
                ["linear_attenuation_pair", "lin_mu_total"],
                lambda: njit_sum(
                    perm_calc.linear_attenuation_pair / perm_calc.lin_mu_total
                ),
            )
            index += 1

            compute_value(
                "p_pair_mean",
                ["linear_attenuation_pair", "lin_mu_total"],
                lambda: njit_mean(
                    perm_calc.linear_attenuation_pair / perm_calc.lin_mu_total
                ),
            )
            index += 1

            compute_value(
                "p_pair_max",
                ["linear_attenuation_pair", "lin_mu_total"],
                lambda: njit_max(
                    perm_calc.linear_attenuation_pair / perm_calc.lin_mu_total
                ),
            )
            index += 1

            compute_value(
                "p_pair_min",
                ["linear_attenuation_pair", "lin_mu_total"],
                lambda: njit_min(
                    perm_calc.linear_attenuation_pair / perm_calc.lin_mu_total
                ),
            )
            index += 1

            compute_value(
                "-log_p_pair_sum",
                ["linear_attenuation_pair", "lin_mu_total"],
                lambda: njit_sum(
                    -np.log(perm_calc.linear_attenuation_pair / perm_calc.lin_mu_total)
                ),
            )
            index += 1

            compute_value(
                "-log_p_pair_mean",
                ["linear_attenuation_pair", "lin_mu_total"],
                lambda: njit_mean(
                    -np.log(perm_calc.linear_attenuation_pair / perm_calc.lin_mu_total)
                ),
            )
            index += 1

            compute_value(
                "-log_p_pair_max",
                ["linear_attenuation_pair", "lin_mu_total"],
                lambda: njit_max(
                    -np.log(perm_calc.linear_attenuation_pair / perm_calc.lin_mu_total)
                ),
            )
            index += 1

            compute_value(
                "-log_p_pair_min",
                ["linear_attenuation_pair", "lin_mu_total"],
                lambda: njit_min(
                    -np.log(perm_calc.linear_attenuation_pair / perm_calc.lin_mu_total)
                ),
            )
            index += 1

    if compute_mode and not njit_any(boolean_vector[index : index + 12]):
        index += 12
    else:
        compute_value(
            "cross_total_sum",
            ["lin_mu_total"],
            lambda: njit_sum(perm_calc.lin_mu_total),
        )
        index += 1

        compute_value(
            "cross_total_mean",
            ["lin_mu_total"],
            lambda: njit_mean(perm_calc.lin_mu_total),
        )
        index += 1

        compute_value(
            "cross_total_max",
            ["lin_mu_total"],
            lambda: njit_max(perm_calc.lin_mu_total),
        )
        index += 1

        compute_value(
            "cross_total_ge_dist_sum",
            ["lin_mu_total"],
            lambda: njit_sum(perm_calc.lin_mu_total * perm_calc.ge_distance_perm),
        )
        index += 1

        compute_value(
            "cross_total_ge_dist_mean",
            ["lin_mu_total"],
            lambda: njit_mean(perm_calc.lin_mu_total * perm_calc.ge_distance_perm),
        )
        index += 1

        compute_value(
            "cross_total_ge_dist_max",
            ["lin_mu_total"],
            lambda: njit_max(perm_calc.lin_mu_total * perm_calc.ge_distance_perm),
        )
        index += 1

        compute_value(
            "cross_total_dist_sum",
            ["lin_mu_total"],
            lambda: njit_sum(perm_calc.lin_mu_total * perm_calc.distance_perm),
        )
        index += 1

        compute_value(
            "cross_total_dist_mean",
            ["lin_mu_total"],
            lambda: njit_mean(perm_calc.lin_mu_total * perm_calc.distance_perm),
        )
        index += 1

        compute_value(
            "cross_total_dist_max",
            ["lin_mu_total"],
            lambda: njit_max(perm_calc.lin_mu_total * perm_calc.distance_perm),
        )
        index += 1

        compute_value(
            "cross_total_min",
            ["lin_mu_total"],
            lambda: njit_min(perm_calc.lin_mu_total),
        )
        index += 1

        compute_value(
            "cross_total_ge_dist_min",
            ["lin_mu_total"],
            lambda: njit_min(perm_calc.lin_mu_total * perm_calc.ge_distance_perm),
        )
        index += 1

        compute_value(
            "cross_total_dist_min",
            ["lin_mu_total"],
            lambda: njit_min(perm_calc.lin_mu_total * perm_calc.distance_perm),
        )
        index += 1

    # %% Klein Nishina features

    if compute_mode and not njit_any(boolean_vector[index : index + 32]):
        index += 32
    else:
        compute_value(
            "klein-nishina_rel_sum_sum",
            ["klein_nishina_relative_use_Ei"],
            lambda: njit_sum(perm_calc.klein_nishina_relative_use_Ei),
        )
        index += 1

        compute_value(
            "klein-nishina_rel_sum_mean",
            ["klein_nishina_relative_use_Ei"],
            lambda: njit_mean(perm_calc.klein_nishina_relative_use_Ei),
        )
        index += 1

        compute_value(
            "klein-nishina_rel_sum_max",
            ["klein_nishina_relative_use_Ei"],
            lambda: njit_max(perm_calc.klein_nishina_relative_use_Ei),
        )
        index += 1

        compute_value(
            "klein-nishina_rel_sum_min",
            ["klein_nishina_relative_use_Ei"],
            lambda: njit_min(perm_calc.klein_nishina_relative_use_Ei),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_rel_sum_sum",
            ["klein_nishina_relative_use_Ei"],
            lambda: njit_sum(-np.log(perm_calc.klein_nishina_relative_use_Ei)),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_rel_sum_mean",
            ["klein_nishina_relative_use_Ei"],
            lambda: njit_mean(-np.log(perm_calc.klein_nishina_relative_use_Ei)),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_rel_sum_max",
            ["klein_nishina_relative_use_Ei"],
            lambda: njit_max(-np.log(perm_calc.klein_nishina_relative_use_Ei)),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_rel_sum_min",
            ["klein_nishina_relative_use_Ei"],
            lambda: njit_min(-np.log(perm_calc.klein_nishina_relative_use_Ei)),
        )
        index += 1

        compute_value(
            "klein-nishina_rel_geo_sum",
            ["klein_nishina_relative"],
            lambda: njit_sum(perm_calc.klein_nishina_relative),
        )
        index += 1

        compute_value(
            "klein-nishina_rel_geo_mean",
            ["klein_nishina_relative"],
            lambda: njit_mean(perm_calc.klein_nishina_relative),
        )
        index += 1

        compute_value(
            "klein-nishina_rel_geo_max",
            ["klein_nishina_relative"],
            lambda: njit_max(perm_calc.klein_nishina_relative),
        )
        index += 1

        compute_value(
            "klein-nishina_rel_geo_min",
            ["klein_nishina_relative"],
            lambda: njit_min(perm_calc.klein_nishina_relative),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_rel_geo_sum",
            ["klein_nishina_relative"],
            lambda: njit_sum(-np.log(perm_calc.klein_nishina_relative)),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_rel_geo_mean",
            ["klein_nishina_relative"],
            lambda: njit_mean(-np.log(perm_calc.klein_nishina_relative)),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_rel_geo_max",
            ["klein_nishina_relative"],
            lambda: njit_max(-np.log(perm_calc.klein_nishina_relative)),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_rel_geo_min",
            ["klein_nishina_relative"],
            lambda: njit_min(-np.log(perm_calc.klein_nishina_relative)),
        )
        index += 1

        compute_value(
            "klein-nishina_sum_sum",
            ["klein_nishina_use_Ei"],
            lambda: njit_sum(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS),
        )
        index += 1

        compute_value(
            "klein-nishina_sum_mean",
            ["klein_nishina_use_Ei"],
            lambda: njit_mean(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS),
        )
        index += 1

        compute_value(
            "klein-nishina_sum_max",
            ["klein_nishina_use_Ei"],
            lambda: njit_max(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS),
        )
        index += 1

        compute_value(
            "klein-nishina_sum_min",
            ["klein_nishina_use_Ei"],
            lambda: njit_min(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_sum_sum",
            ["klein_nishina_use_Ei"],
            lambda: njit_sum(-np.log(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS)),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_sum_mean",
            ["klein_nishina_use_Ei"],
            lambda: njit_mean(-np.log(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS)),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_sum_max",
            ["klein_nishina_use_Ei"],
            lambda: njit_max(-np.log(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS)),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_sum_min",
            ["klein_nishina_use_Ei"],
            lambda: njit_min(-np.log(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS)),
        )
        index += 1

        compute_value(
            "klein-nishina_geo_sum",
            ["klein_nishina"],
            lambda: njit_sum(perm_calc.klein_nishina * RANGE_PROCESS),
        )
        index += 1

        compute_value(
            "klein-nishina_geo_mean",
            ["klein_nishina"],
            lambda: njit_mean(perm_calc.klein_nishina * RANGE_PROCESS),
        )
        index += 1

        compute_value(
            "klein-nishina_geo_max",
            ["klein_nishina"],
            lambda: njit_max(perm_calc.klein_nishina * RANGE_PROCESS),
        )
        index += 1

        compute_value(
            "klein-nishina_geo_min",
            ["klein_nishina"],
            lambda: njit_min(perm_calc.klein_nishina * RANGE_PROCESS),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_geo_sum",
            ["klein_nishina"],
            lambda: njit_sum(-np.log(perm_calc.klein_nishina * RANGE_PROCESS)),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_geo_mean",
            ["klein_nishina"],
            lambda: njit_mean(-np.log(perm_calc.klein_nishina * RANGE_PROCESS)),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_geo_max",
            ["klein_nishina"],
            lambda: njit_max(-np.log(perm_calc.klein_nishina * RANGE_PROCESS)),
        )
        index += 1

        compute_value(
            "-log_klein-nishina_geo_min",
            ["klein_nishina"],
            lambda: njit_min(-np.log(perm_calc.klein_nishina * RANGE_PROCESS)),
        )
        index += 1

    # %% formerly cluster property features

    if compute_mode and not njit_any(boolean_vector[index : index + 12]):
        index += 12
    else:
        compute_value("first_r", ["radii_perm"], lambda: perm_calc.radii_perm[0])
        index += 1

        compute_value("final_r", ["radii_perm"], lambda: perm_calc.radii_perm[-1])
        index += 1

        compute_value(
            "first_energy_ratio",
            ["energies_perm", "energy_sum"],
            lambda: perm_calc.energies_perm[0] / perm_calc.energy_sum,
        )
        index += 1

        compute_value(
            "final_energy_ratio",
            ["energies_perm"],
            lambda: perm_calc.energies_perm[-2]
            / (perm_calc.energies_perm[-2] + perm_calc.energies_perm[-1]),
        )
        index += 1

        compute_value(
            "first_is_not_largest",
            ["energies_perm"],
            lambda: njit_any(perm_calc.energies_perm[1:] > perm_calc.energies_perm[0]),
        )
        index += 1

        compute_value(
            "first_is_not_closest",
            ["radii_perm"],
            lambda: njit_any(perm_calc.radii_perm[1:] < perm_calc.radii_perm[0]),
        )
        index += 1

        compute_value(
            "tango_variance",
            ["tango_estimates_perm"],
            lambda: np.var(perm_calc.tango_estimates_perm),
        )
        index += 1

        compute_value(
            "tango_v_variance",
            ["tango_estimates_sigma_perm"],
            lambda: 1.0 / njit_sum(1.0 / perm_calc.tango_estimates_sigma_perm**2),
        )
        index += 1

        compute_value(
            "tango_sigma",
            ["tango_estimates_perm"],
            lambda: np.std(perm_calc.tango_estimates_perm),
        )
        index += 1

        compute_value(
            "tango_v_sigma",
            ["tango_estimates_sigma_perm"],
            lambda: np.sqrt(
                1.0 / njit_sum(1.0 / perm_calc.tango_estimates_sigma_perm**2)
            ),
        )
        index += 1

        compute_value(
            "escape_probability",
            ["escape_probability"],
            lambda: perm_calc.escape_probability,
        )
        index += 1

        compute_value(
            "-log_escape_probability",
            ["escape_probability"],
            lambda: -np.log(perm_calc.escape_probability + 1e-16),
        )
        index += 1

    if name_mode:
        return names
    if dependency_mode:
        return dependencies_dict
    return features_vector
