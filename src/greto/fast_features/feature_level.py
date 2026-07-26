"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Feature level computations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

__all__ = ["FeatureSpec", "build_feature_specs", "feature_values"]

import numba
import numpy as np
import functools

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


# --- Pure helper functions (top-level, easy to JIT later) ------------------
def sum_attr(perm_calc, attr, Nmi=None):
    return njit_sum(getattr(perm_calc, attr))


def mean_attr(perm_calc, attr, Nmi=None):
    return njit_mean(getattr(perm_calc, attr))


def norm_div_Nmi(perm_calc, attr, Nmi):
    return njit_norm(getattr(perm_calc, attr)) / Nmi


def sum_sq_attr(perm_calc, attr, Nmi=None):
    return njit_sum(getattr(perm_calc, attr) ** 2)


def mean_sq_attr(perm_calc, attr, Nmi=None):
    return njit_mean(getattr(perm_calc, attr) ** 2)


def first_elem(perm_calc, attr, Nmi=None):
    return getattr(perm_calc, attr)[0]


def last_elem(perm_calc, attr, Nmi=None):
    return getattr(perm_calc, attr)[-1]


def nth_elem(perm_calc, attr, n, Nmi=None):
    return getattr(perm_calc, attr)[n]


def wmean_from_attrs(perm_calc, v_attr, sigma_attr, Nmi=None):
    return wmean_1v_func(getattr(perm_calc, v_attr), getattr(perm_calc, sigma_attr))


def wmean2_from_attrs(perm_calc, v_attr, sigma_attr, Nmi=None):
    return wmean_2v_func(getattr(perm_calc, v_attr), getattr(perm_calc, sigma_attr))

# ---------------------------------------------------------------------------

# --- More generic pure helpers --------------------------------------------
def min_attr(perm_calc, attr, Nmi=None):
    return njit_min(getattr(perm_calc, attr))


def max_attr(perm_calc, attr, Nmi=None):
    return njit_max(getattr(perm_calc, attr))


def sum_mul_attrs(perm_calc, a1, a2, Nmi=None):
    return njit_sum(getattr(perm_calc, a1) * getattr(perm_calc, a2))


def mean_mul_attrs(perm_calc, a1, a2, Nmi=None):
    return njit_mean(getattr(perm_calc, a1) * getattr(perm_calc, a2))


def sum_mul_attrs_nonfinal(perm_calc, a1, a2, Nmi=None):
    return njit_sum(getattr(perm_calc, a1)[:-1] * getattr(perm_calc, a2)[:-1])


def mean_mul_attrs_nonfinal(perm_calc, a1, a2, Nmi=None):
    return njit_mean(getattr(perm_calc, a1)[:-1] * getattr(perm_calc, a2)[:-1])


def sum_nonfinal(perm_calc, attr, Nmi=None):
    return njit_sum(getattr(perm_calc, attr)[:-1])


def mean_nonfinal(perm_calc, attr, Nmi=None):
    return njit_mean(getattr(perm_calc, attr)[:-1])


def min_nonfinal(perm_calc, attr, Nmi=None):
    return njit_min(getattr(perm_calc, attr)[:-1])


def max_nonfinal(perm_calc, attr, Nmi=None):
    return njit_max(getattr(perm_calc, attr)[:-1])


def sum_div_attrs(perm_calc, a1, a2, Nmi=None):
    return njit_sum(getattr(perm_calc, a1) / getattr(perm_calc, a2))


def mean_div_attrs(perm_calc, a1, a2, Nmi=None):
    return njit_mean(getattr(perm_calc, a1) / getattr(perm_calc, a2))


def neglog_sum_div(perm_calc, a1, a2, Nmi=None):
    return njit_sum(-np.log(getattr(perm_calc, a1) / getattr(perm_calc, a2)))


def neglog_mean_div(perm_calc, a1, a2, Nmi=None):
    return njit_mean(-np.log(getattr(perm_calc, a1) / getattr(perm_calc, a2)))


def neglog_max_div(perm_calc, a1, a2, Nmi=None):
    return njit_max(-np.log(getattr(perm_calc, a1) / getattr(perm_calc, a2)))


def sum_mul_attr_scalar(perm_calc, attr, scalar, Nmi=None):
    return njit_sum(getattr(perm_calc, attr) * scalar)


def mean_mul_attr_scalar(perm_calc, attr, scalar, Nmi=None):
    return njit_mean(getattr(perm_calc, attr) * scalar)

# ---------------------------------------------------------------------------

def sum_mul_attr_one_minus_attr(perm_calc, a, b, Nmi=None):
    return njit_sum(getattr(perm_calc, a) * (1.0 - getattr(perm_calc, b)))


def mean_mul_attr_one_minus_attr(perm_calc, a, b, Nmi=None):
    return njit_mean(getattr(perm_calc, a) * (1.0 - getattr(perm_calc, b)))


def sum_sq_mul_attr_one_minus_attr(perm_calc, a, b, Nmi=None):
    return njit_sum((getattr(perm_calc, a) ** 2) * (1.0 - getattr(perm_calc, b)))


def mean_sq_mul_attr_one_minus_attr(perm_calc, a, b, Nmi=None):
    return njit_mean((getattr(perm_calc, a) ** 2) * (1.0 - getattr(perm_calc, b)))


def rc_wmean_1v_from_attrs(perm_calc, penalty_attr, v_attr, sigma_attr, Nmi=None):
    return rc_wmean_1v_penalty_removed_func(
        getattr(perm_calc, penalty_attr), getattr(perm_calc, v_attr), getattr(perm_calc, sigma_attr)
    )


def rc_wmean_2v_from_attrs(perm_calc, penalty_attr, v_attr, sigma_attr, Nmi=None):
    return rc_wmean_2v_penalty_removed_func(
        getattr(perm_calc, penalty_attr), getattr(perm_calc, v_attr), getattr(perm_calc, sigma_attr)
    )


def rth_wmean_1v_from_attrs(perm_calc, penalty_attr, v_attr, sigma_attr, Nmi=None):
    return rth_wmean_1v_penalty_removed_func(
        getattr(perm_calc, penalty_attr), getattr(perm_calc, v_attr), getattr(perm_calc, sigma_attr)
    )


def rth_wmean_2v_from_attrs(perm_calc, penalty_attr, v_attr, sigma_attr, Nmi=None):
    return rth_wmean_2v_penalty_removed_func(
        getattr(perm_calc, penalty_attr), getattr(perm_calc, v_attr), getattr(perm_calc, sigma_attr)
    )

# ---------------------------------------------------------------------------
def last_div_attrs(perm_calc, a1, a2, Nmi=None):
    return getattr(perm_calc, a1)[-1] / getattr(perm_calc, a2)[-1]


def neglog_last_div(perm_calc, a1, a2, Nmi=None):
    return -np.log(last_div_attrs(perm_calc, a1, a2))


def sum_div_attrs_nonfinal(perm_calc, a1, a2, Nmi=None):
    return njit_sum(getattr(perm_calc, a1)[:-1] / getattr(perm_calc, a2)[:-1])


def mean_div_attrs_nonfinal(perm_calc, a1, a2, Nmi=None):
    return njit_mean(getattr(perm_calc, a1)[:-1] / getattr(perm_calc, a2)[:-1])


def neglog_sum_div_nonfinal(perm_calc, a1, a2, Nmi=None):
    return njit_sum(-np.log(getattr(perm_calc, a1)[:-1] / getattr(perm_calc, a2)[:-1]))


def neglog_mean_div_nonfinal(perm_calc, a1, a2, Nmi=None):
    return njit_mean(-np.log(getattr(perm_calc, a1)[:-1] / getattr(perm_calc, a2)[:-1]))


def neglog_min_div_nonfinal(perm_calc, a1, a2, Nmi=None):
    return njit_min(-np.log(getattr(perm_calc, a1)[:-1] / getattr(perm_calc, a2)[:-1]))


def neglog_sum_mul_attr_scalar(perm_calc, attr, scalar, Nmi=None):
    return njit_sum(-np.log(getattr(perm_calc, attr) * scalar))


def neglog_mean_mul_attr_scalar(perm_calc, attr, scalar, Nmi=None):
    return njit_mean(-np.log(getattr(perm_calc, attr) * scalar))


def any_greater_slice_first(perm_calc, attr, Nmi=None):
    return njit_any(getattr(perm_calc, attr)[1:] > getattr(perm_calc, attr)[0])


def any_less_slice_first(perm_calc, attr, Nmi=None):
    return njit_any(getattr(perm_calc, attr)[1:] < getattr(perm_calc, attr)[0])


def var_attr(perm_calc, attr, Nmi=None):
    return np.var(getattr(perm_calc, attr))


def std_attr(perm_calc, attr, Nmi=None):
    return np.std(getattr(perm_calc, attr))


def inv_sum_inv_sq(perm_calc, attr, Nmi=None):
    return 1.0 / njit_sum(1.0 / getattr(perm_calc, attr) ** 2)


def neglog_scalar_attr(perm_calc, attr, eps=1e-16, Nmi=None):
    return -np.log(getattr(perm_calc, attr) + eps)

# ---------------------------------------------------------------------------
def norm_div_Nmi_sqrtlen(perm_calc, attr, Nmi=None):
    return njit_norm(getattr(perm_calc, attr)) / Nmi / np.sqrt(len(getattr(perm_calc, attr)))


def max_div_attrs(perm_calc, a1, a2, Nmi=None):
    return njit_max(getattr(perm_calc, a1) / getattr(perm_calc, a2))


def min_div_attrs(perm_calc, a1, a2, Nmi=None):
    return njit_min(getattr(perm_calc, a1) / getattr(perm_calc, a2))


def neglog_min_div(perm_calc, a1, a2, Nmi=None):
    return njit_min(-np.log(getattr(perm_calc, a1) / getattr(perm_calc, a2)))


def first_div_by_scalar(perm_calc, arr_attr, scalar_attr, Nmi=None):
    return getattr(perm_calc, arr_attr)[0] / getattr(perm_calc, scalar_attr)


def final_pair_ratio(perm_calc, arr_attr, Nmi=None):
    arr = getattr(perm_calc, arr_attr)
    return arr[-2] / (arr[-2] + arr[-1])


def last_mul_attrs(perm_calc, a1, a2, Nmi=None):
    return getattr(perm_calc, a1)[-1] * getattr(perm_calc, a2)[-1]


def max_mul_attrs(perm_calc, a1, a2, Nmi=None):
    return njit_max(getattr(perm_calc, a1) * getattr(perm_calc, a2))


def min_mul_attrs(perm_calc, a1, a2, Nmi=None):
    return njit_min(getattr(perm_calc, a1) * getattr(perm_calc, a2))


def min_mul_attrs_nonfinal(perm_calc, a1, a2, Nmi=None):
    return njit_min(getattr(perm_calc, a1)[:-1] * getattr(perm_calc, a2)[:-1])


def max_mul_attrs_nonfinal(perm_calc, a1, a2, Nmi=None):
    return njit_max(getattr(perm_calc, a1)[:-1] * getattr(perm_calc, a2)[:-1])

# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dependencies: tuple[str, ...]
    compute_fn: Optional[Callable[[], float]] = None


def build_feature_specs(perm_calc=None, Nmi=None):
    specs = []

    def add_feature(name, dependencies, compute_fn):
        # If a perm_calc is provided, ensure compute_fn is a zero-arg callable
        wrapped = None
        if compute_fn is None:
            wrapped = None
        elif perm_calc is None:
            # name/dependency mode only; keep as-is
            wrapped = compute_fn
        else:
            def _bound(fn=compute_fn, pc=perm_calc, nm=Nmi):
                try:
                    return fn(pc, nm)
                except TypeError:
                    try:
                        return fn(pc)
                    except TypeError:
                        return fn()

            wrapped = _bound

        specs.append(FeatureSpec(name=name, dependencies=tuple(dependencies), compute_fn=wrapped))

    _append_residual_geo_features(add_feature, perm_calc, Nmi)
    _append_residual_loc_features(add_feature, perm_calc, Nmi)
    _append_residual_loc_geo_features(add_feature, perm_calc, Nmi)
    _append_compton_penalty_features(add_feature, perm_calc)
    _append_cos_features(add_feature, perm_calc, Nmi)
    _append_cos_cap_features(add_feature, perm_calc, Nmi)
    _append_theta_features(add_feature, perm_calc, Nmi)
    _append_theta_cap_features(add_feature, perm_calc, Nmi)
    _append_distance_and_cross_section_features(add_feature, perm_calc)
    _append_probability_features(add_feature, perm_calc)
    _append_total_cross_features(add_feature, perm_calc)
    _append_klein_nishina_features(add_feature, perm_calc)
    _append_cluster_property_features(add_feature, perm_calc)

    return specs


def _append_residual_geo_features(add_feature, perm_calc, Nmi):
    add_feature("rsg_sum_1", ["res_sum_geo"], functools.partial(sum_attr, attr="res_sum_geo"))
    add_feature(
        "rsg_sum_1_first", ["res_sum_geo"], functools.partial(first_elem, attr="res_sum_geo")
    )
    add_feature("rsg_mean_1", ["res_sum_geo"], functools.partial(mean_attr, attr="res_sum_geo"))
    add_feature(
        "rsg_mean_1_first",
        ["res_sum_geo"],
        functools.partial(nth_elem, attr="res_sum_geo", n=0),
    )
    add_feature(
        "rsg_wmean_1v",
        ["res_sum_geo_v", "res_sum_geo_sigma"],
        functools.partial(wmean_from_attrs, v_attr="res_sum_geo_v", sigma_attr="res_sum_geo_sigma"),
    )
    add_feature(
        "rsg_wmean_1v_first",
        ["res_sum_geo_v", "res_sum_geo_sigma"],
        functools.partial(nth_elem, attr="res_sum_geo_v", n=0),
    )
    add_feature("rsg_norm_2", ["res_sum_geo"], functools.partial(norm_div_Nmi, attr="res_sum_geo"))
    add_feature("rsg_sum_2", ["res_sum_geo"], functools.partial(sum_sq_attr, attr="res_sum_geo"))
    add_feature(
        "rsg_sum_2_first", ["res_sum_geo"], functools.partial(nth_elem, attr="res_sum_geo", n=0)
    )
    add_feature("rsg_mean_2", ["res_sum_geo"], functools.partial(mean_sq_attr, attr="res_sum_geo"))
    add_feature(
        "rsg_mean_2_first",
        ["res_sum_geo"],
        functools.partial(lambda pc, attr, Nmi=None: nth_elem(pc, attr, 0) ** 2, attr="res_sum_geo"),
    )
    add_feature(
        "rsg_wmean_2v",
        ["res_sum_geo_v", "res_sum_geo_sigma"],
        functools.partial(wmean2_from_attrs, v_attr="res_sum_geo_v", sigma_attr="res_sum_geo_sigma"),
    )
    add_feature(
        "rsg_wmean_2v_first",
        ["res_sum_geo_v", "res_sum_geo_sigma"],
        functools.partial(lambda pc, attr, Nmi=None: nth_elem(pc, attr, 0) ** 2, attr="res_sum_geo_v"),
    )
    add_feature("rsg_sum_1v", ["res_sum_geo_v"], functools.partial(sum_attr, attr="res_sum_geo_v"))
    add_feature("rsg_sum_1v_first", ["res_sum_geo_v"], functools.partial(first_elem, attr="res_sum_geo_v"))
    add_feature("rsg_mean_1v", ["res_sum_geo_v"], functools.partial(mean_attr, attr="res_sum_geo_v"))
    add_feature(
        "rsg_mean_1v_first",
        ["res_sum_geo_v"],
        functools.partial(nth_elem, attr="res_sum_geo_v", n=0),
    )
    add_feature("rsg_norm_2v", ["res_sum_geo_v"], functools.partial(norm_div_Nmi, attr="res_sum_geo_v"))
    add_feature("rsg_sum_2v", ["res_sum_geo_v"], functools.partial(sum_sq_attr, attr="res_sum_geo_v"))
    add_feature(
        "rsg_sum_2v_first", ["res_sum_geo_v"], functools.partial(lambda pc, attr, Nmi=None: nth_elem(pc, attr, 0) ** 2, attr="res_sum_geo_v")
    )
    add_feature("rsg_mean_2v", ["res_sum_geo_v"], functools.partial(mean_sq_attr, attr="res_sum_geo_v"))
    add_feature(
        "rsg_mean_2v_first",
        ["res_sum_geo_v"],
        functools.partial(lambda pc, attr, Nmi=None: nth_elem(pc, attr, 0) ** 2, attr="res_sum_geo_v"),
    )


def _append_residual_loc_features(add_feature, perm_calc, Nmi):
    add_feature("rsl_mean_1", ["res_sum_loc"], functools.partial(mean_attr, attr="res_sum_loc"))
    add_feature("rsl_sum_1", ["res_sum_loc"], functools.partial(sum_attr, attr="res_sum_loc"))
    add_feature("rsl_norm_2", ["res_sum_loc"], functools.partial(norm_div_Nmi, attr="res_sum_loc"))
    add_feature("rsl_sum_2", ["res_sum_loc"], functools.partial(sum_sq_attr, attr="res_sum_loc"))
    add_feature("rsl_mean_2", ["res_sum_loc"], functools.partial(mean_sq_attr, attr="res_sum_loc"))
    add_feature("rsl_sum_1v", ["res_sum_loc_v"], functools.partial(sum_attr, attr="res_sum_loc_v"))
    add_feature("rsl_mean_1v", ["res_sum_loc_v"], functools.partial(mean_attr, attr="res_sum_loc_v"))
    add_feature(
        "rsl_norm_2v",
        ["res_sum_loc_v"],
        functools.partial(lambda pc, attr, Nmi=None: njit_norm(getattr(pc, attr)) / Nmi / np.sqrt(len(getattr(pc, attr))), attr="res_sum_loc_v"),
    )
    add_feature("rsl_mean_2v", ["res_sum_loc_v"], functools.partial(mean_sq_attr, attr="res_sum_loc_v"))
    add_feature("rsl_sum_2v", ["res_sum_loc_v"], functools.partial(sum_sq_attr, attr="res_sum_loc_v"))
    add_feature(
        "rsl_wmean_2v",
        ["res_sum_loc_v", "res_sum_loc_sigma"],
        functools.partial(wmean2_from_attrs, v_attr="res_sum_loc_v", sigma_attr="res_sum_loc_sigma"),
    )
    add_feature(
        "rsl_wmean_1v",
        ["res_sum_loc_v", "res_sum_loc_sigma"],
        functools.partial(wmean_from_attrs, v_attr="res_sum_loc_v", sigma_attr="res_sum_loc_sigma"),
    )


def _append_residual_loc_geo_features(add_feature, perm_calc, Nmi):
    add_feature("rlg_sum_1v", ["res_loc_geo_v"], functools.partial(sum_attr, attr="res_loc_geo_v"))
    add_feature("rlg_mean_1v", ["res_loc_geo_v"], functools.partial(mean_attr, attr="res_loc_geo_v"))
    add_feature(
        "rlg_norm_2v",
        ["res_loc_geo_v"],
        functools.partial(norm_div_Nmi_sqrtlen, attr="res_loc_geo_v"),
    )
    add_feature("rlg_sum_2v", ["res_loc_geo_v"], functools.partial(sum_sq_attr, attr="res_loc_geo_v"))
    add_feature("rlg_mean_2v", ["res_loc_geo_v"], functools.partial(mean_sq_attr, attr="res_loc_geo_v"))
    add_feature("rlg_sum_1", ["res_loc_geo"], functools.partial(sum_attr, attr="res_loc_geo"))
    add_feature("rlg_mean_1", ["res_loc_geo"], functools.partial(mean_attr, attr="res_loc_geo"))
    add_feature("rlg_norm_2", ["res_loc_geo"], functools.partial(norm_div_Nmi, attr="res_loc_geo"))
    add_feature(
        "rlg_wmean_1v",
        ["res_loc_geo_v", "res_loc_geo_sigma"],
        functools.partial(wmean_from_attrs, v_attr="res_loc_geo_v", sigma_attr="res_loc_geo_sigma"),
    )
    add_feature("rlg_sum_2", ["res_loc_geo"], functools.partial(sum_sq_attr, attr="res_loc_geo"))
    add_feature("rlg_mean_2", ["res_loc_geo"], functools.partial(mean_sq_attr, attr="res_loc_geo"))
    add_feature(
        "rlg_wmean_2v",
        ["res_loc_geo_v", "res_loc_geo_sigma"],
        functools.partial(wmean2_from_attrs, v_attr="res_loc_geo_v", sigma_attr="res_loc_geo_sigma"),
    )


def _append_compton_penalty_features(add_feature, perm_calc):
    add_feature("c_penalty_sum_1", ["compton_penalty"], lambda: njit_sum(perm_calc.compton_penalty))
    add_feature("c_penalty_mean_1", ["compton_penalty"], lambda: njit_mean(perm_calc.compton_penalty))
    add_feature(
        "c_penalty_ell_sum_1",
        ["compton_penalty_ell1"],
        lambda: njit_sum(perm_calc.compton_penalty_ell1),
    )
    add_feature(
        "c_penalty_ell_mean_1",
        ["compton_penalty_ell1"],
        lambda: njit_mean(perm_calc.compton_penalty_ell1),
    )
    add_feature(
        "c_penalty_ell_sum_2",
        ["compton_penalty_ell1"],
        lambda: njit_sum(perm_calc.compton_penalty_ell1**2),
    )
    add_feature(
        "c_penalty_ell_mean_2",
        ["compton_penalty_ell1"],
        lambda: njit_mean(perm_calc.compton_penalty_ell1**2),
    )


def _append_cos_features(add_feature, perm_calc, Nmi):
    add_feature("rc_sum_1", ["res_cos"], functools.partial(sum_attr, attr="res_cos"))
    add_feature("rc_mean_1", ["res_cos"], functools.partial(mean_attr, attr="res_cos"))
    add_feature("rc_norm_2", ["res_cos"], functools.partial(norm_div_Nmi, attr="res_cos"))
    add_feature("rc_sum_2", ["res_cos"], functools.partial(sum_sq_attr, attr="res_cos"))
    add_feature("rc_mean_2", ["res_cos"], functools.partial(mean_sq_attr, attr="res_cos"))
    add_feature(
        "rc_sum_1_penalty_removed",
        ["res_cos", "compton_penalty"],
        functools.partial(sum_mul_attr_one_minus_attr, a="res_cos", b="compton_penalty"),
    )
    add_feature(
        "rc_mean_1_penalty_removed",
        ["res_cos", "compton_penalty"],
        functools.partial(mean_mul_attr_one_minus_attr, a="res_cos", b="compton_penalty"),
    )
    add_feature(
        "rc_sum_2_penalty_removed",
        ["res_cos", "compton_penalty"],
        functools.partial(sum_sq_mul_attr_one_minus_attr, a="res_cos", b="compton_penalty"),
    )
    add_feature(
        "rc_mean_2_penalty_removed",
        ["res_cos", "compton_penalty"],
        functools.partial(mean_sq_mul_attr_one_minus_attr, a="res_cos", b="compton_penalty"),
    )
    add_feature(
        "rc_wmean_1v",
        ["res_cos_v", "res_cos_sigma"],
        functools.partial(wmean_from_attrs, v_attr="res_cos_v", sigma_attr="res_cos_sigma"),
    )
    add_feature(
        "rc_wmean_2v",
        ["res_cos_v", "res_cos_sigma"],
        functools.partial(wmean2_from_attrs, v_attr="res_cos_v", sigma_attr="res_cos_sigma"),
    )
    add_feature("rc_sum_1v", ["res_cos_v"], functools.partial(sum_attr, attr="res_cos_v"))
    add_feature("rc_mean_1v", ["res_cos_v"], functools.partial(mean_attr, attr="res_cos_v"))
    add_feature("rc_norm_2v", ["res_cos_v"], functools.partial(norm_div_Nmi, attr="res_cos_v"))
    add_feature("rc_sum_2v", ["res_cos_v"], functools.partial(sum_sq_attr, attr="res_cos_v"))
    add_feature("rc_mean_2v", ["res_cos_v"], functools.partial(mean_sq_attr, attr="res_cos_v"))
    add_feature(
        "rc_wmean_1v_penalty_removed",
        ["compton_penalty", "res_cos_v", "res_cos_sigma"],
        functools.partial(rc_wmean_1v_from_attrs, penalty_attr="compton_penalty", v_attr="res_cos_v", sigma_attr="res_cos_sigma"),
    )
    add_feature(
        "rc_wmean_2v_penalty_removed",
        ["compton_penalty", "res_cos_v", "res_cos_sigma"],
        functools.partial(rc_wmean_2v_from_attrs, penalty_attr="compton_penalty", v_attr="res_cos_v", sigma_attr="res_cos_sigma"),
    )
    add_feature(
        "rc_sum_1v_penalty_removed",
        ["res_cos_v", "compton_penalty"],
        functools.partial(sum_mul_attr_one_minus_attr, a="res_cos_v", b="compton_penalty"),
    )
    add_feature(
        "rc_mean_1v_penalty_removed",
        ["res_cos_v", "compton_penalty"],
        functools.partial(mean_mul_attr_one_minus_attr, a="res_cos_v", b="compton_penalty"),
    )
    add_feature(
        "rc_sum_2v_penalty_removed",
        ["res_cos_v", "compton_penalty"],
        functools.partial(sum_sq_mul_attr_one_minus_attr, a="res_cos_v", b="compton_penalty"),
    )
    add_feature(
        "rc_mean_2v_penalty_removed",
        ["res_cos_v", "compton_penalty"],
        functools.partial(mean_sq_mul_attr_one_minus_attr, a="res_cos_v", b="compton_penalty"),
    )


def _append_cos_cap_features(add_feature, perm_calc, Nmi):
    add_feature("rc_cap_sum_1", ["res_cos_cap"], functools.partial(sum_attr, attr="res_cos_cap"))
    add_feature("rc_cap_mean_1", ["res_cos_cap"], functools.partial(mean_attr, attr="res_cos_cap"))
    add_feature("rc_cap_norm_2", ["res_cos_cap"], functools.partial(norm_div_Nmi, attr="res_cos_cap"))
    add_feature("rc_cap_sum_2", ["res_cos_cap"], functools.partial(sum_sq_attr, attr="res_cos_cap"))
    add_feature("rc_cap_mean_2", ["res_cos_cap"], functools.partial(mean_sq_attr, attr="res_cos_cap"))
    add_feature(
        "rc_cap_wmean_1v",
        ["res_cos_cap_v", "res_cos_sigma"],
        functools.partial(wmean_from_attrs, v_attr="res_cos_cap_v", sigma_attr="res_cos_sigma"),
    )
    add_feature(
        "rc_cap_wmean_2v",
        ["res_cos_cap_v", "res_cos_sigma"],
        functools.partial(wmean2_from_attrs, v_attr="res_cos_cap_v", sigma_attr="res_cos_sigma"),
    )
    add_feature("rc_cap_sum_1v", ["res_cos_cap_v"], functools.partial(sum_attr, attr="res_cos_cap_v"))
    add_feature("rc_cap_mean_1v", ["res_cos_cap_v"], functools.partial(mean_attr, attr="res_cos_cap_v"))
    add_feature("rc_cap_norm_2v", ["res_cos_cap_v"], functools.partial(norm_div_Nmi, attr="res_cos_cap_v"))
    add_feature("rc_cap_sum_2v", ["res_cos_cap_v"], functools.partial(sum_sq_attr, attr="res_cos_cap_v"))
    add_feature("rc_cap_mean_2v", ["res_cos_cap_v"], functools.partial(mean_sq_attr, attr="res_cos_cap_v"))


def _append_theta_features(add_feature, perm_calc, Nmi):
    add_feature("rth_sum_1", ["res_theta"], functools.partial(sum_attr, attr="res_theta"))
    add_feature("rth_mean_1", ["res_theta"], functools.partial(mean_attr, attr="res_theta"))
    add_feature("rth_norm_2", ["res_theta"], functools.partial(norm_div_Nmi, attr="res_theta"))
    add_feature("rth_sum_2", ["res_theta"], functools.partial(sum_sq_attr, attr="res_theta"))
    add_feature("rth_mean_2", ["res_theta"], functools.partial(mean_sq_attr, attr="res_theta"))
    add_feature(
        "rth_sum_1_penalty_removed",
        ["res_theta", "compton_penalty"],
        functools.partial(sum_mul_attr_one_minus_attr, a="res_theta", b="compton_penalty"),
    )
    add_feature(
        "rth_mean_1_penalty_removed",
        ["res_theta", "compton_penalty"],
        functools.partial(mean_mul_attr_one_minus_attr, a="res_theta", b="compton_penalty"),
    )
    add_feature(
        "rth_sum_2_penalty_removed",
        ["res_theta", "compton_penalty"],
        functools.partial(sum_sq_mul_attr_one_minus_attr, a="res_theta", b="compton_penalty"),
    )
    add_feature(
        "rth_mean_2_penalty_removed",
        ["res_theta", "compton_penalty"],
        functools.partial(mean_sq_mul_attr_one_minus_attr, a="res_theta", b="compton_penalty"),
    )
    add_feature(
        "rth_wmean_1v",
        ["res_theta_v", "res_theta_sigma"],
        functools.partial(wmean_from_attrs, v_attr="res_theta_v", sigma_attr="res_theta_sigma"),
    )
    add_feature(
        "rth_wmean_2v",
        ["res_theta_v", "res_theta_sigma"],
        functools.partial(wmean2_from_attrs, v_attr="res_theta_v", sigma_attr="res_theta_sigma"),
    )
    add_feature("rth_sum_1v", ["res_theta_v"], functools.partial(sum_attr, attr="res_theta_v"))
    add_feature("rth_mean_1v", ["res_theta_v"], functools.partial(mean_attr, attr="res_theta_v"))
    add_feature("rth_norm_2v", ["res_theta_v"], functools.partial(norm_div_Nmi, attr="res_theta_v"))
    add_feature("rth_sum_2v", ["res_theta_v"], functools.partial(sum_sq_attr, attr="res_theta_v"))
    add_feature("rth_mean_2v", ["res_theta_v"], functools.partial(mean_sq_attr, attr="res_theta_v"))
    add_feature(
        "rth_sum_1v_penalty_removed",
        ["res_theta_v", "compton_penalty"],
        functools.partial(sum_mul_attr_one_minus_attr, a="res_theta_v", b="compton_penalty"),
    )
    add_feature(
        "rth_mean_1v_penalty_removed",
        ["res_theta_v", "compton_penalty"],
        functools.partial(mean_mul_attr_one_minus_attr, a="res_theta_v", b="compton_penalty"),
    )
    add_feature(
        "rth_sum_2v_penalty_removed",
        ["res_theta_v", "compton_penalty"],
        functools.partial(sum_sq_mul_attr_one_minus_attr, a="res_theta_v", b="compton_penalty"),
    )
    add_feature(
        "rth_mean_2v_penalty_removed",
        ["res_theta_v", "compton_penalty"],
        functools.partial(mean_sq_mul_attr_one_minus_attr, a="res_theta_v", b="compton_penalty"),
    )
    add_feature(
        "rth_wmean_1v_penalty_removed",
        ["compton_penalty", "res_theta_v", "res_theta_sigma"],
        functools.partial(rth_wmean_1v_from_attrs, penalty_attr="compton_penalty", v_attr="res_theta_v", sigma_attr="res_theta_sigma"),
    )
    add_feature(
        "rth_wmean_2v_penalty_removed",
        ["compton_penalty", "res_theta_v", "res_theta_sigma"],
        functools.partial(rth_wmean_2v_from_attrs, penalty_attr="compton_penalty", v_attr="res_theta_v", sigma_attr="res_theta_sigma"),
    )


def _append_theta_cap_features(add_feature, perm_calc, Nmi):
    add_feature("rth_cap_sum_1", ["res_theta_cap"], functools.partial(sum_attr, attr="res_theta_cap"))
    add_feature("rth_cap_mean_1", ["res_theta_cap"], functools.partial(mean_attr, attr="res_theta_cap"))
    add_feature("rth_cap_norm_2", ["res_theta_cap"], functools.partial(norm_div_Nmi, attr="res_theta_cap"))
    add_feature("rth_cap_sum_2", ["res_theta_cap"], functools.partial(sum_sq_attr, attr="res_theta_cap"))
    add_feature("rth_cap_mean_2", ["res_theta_cap"], functools.partial(mean_sq_attr, attr="res_theta_cap"))
    add_feature(
        "rth_cap_wmean_1v",
        ["res_theta_cap_v", "res_theta_sigma"],
        functools.partial(wmean_from_attrs, v_attr="res_theta_cap_v", sigma_attr="res_theta_sigma"),
    )
    add_feature(
        "rth_cap_wmean_2v",
        ["res_theta_cap_v", "res_theta_sigma"],
        functools.partial(wmean2_from_attrs, v_attr="res_theta_cap_v", sigma_attr="res_theta_sigma"),
    )
    add_feature("rth_cap_sum_1v", ["res_theta_cap_v"], functools.partial(sum_attr, attr="res_theta_cap_v"))
    add_feature("rth_cap_mean_1v", ["res_theta_cap_v"], functools.partial(mean_attr, attr="res_theta_cap_v"))
    add_feature("rth_cap_norm_2v", ["res_theta_cap_v"], functools.partial(norm_div_Nmi, attr="res_theta_cap_v"))
    add_feature("rth_cap_sum_2v", ["res_theta_cap_v"], functools.partial(sum_sq_attr, attr="res_theta_cap_v"))
    add_feature("rth_cap_mean_2v", ["res_theta_cap_v"], functools.partial(mean_sq_attr, attr="res_theta_cap_v"))


def _append_distance_and_cross_section_features(add_feature, perm_calc):
    add_feature("distances_sum", ["distance_perm"], functools.partial(sum_attr, attr="distance_perm"))
    add_feature("distances_mean", ["distance_perm"], functools.partial(mean_attr, attr="distance_perm"))
    add_feature("ge_distances_sum", ["ge_distance_perm"], functools.partial(sum_attr, attr="ge_distance_perm"))
    add_feature("ge_distances_mean", ["ge_distance_perm"], functools.partial(mean_attr, attr="ge_distance_perm"))

    add_feature(
        "cross_abs_sum",
        ["linear_attenuation_abs"],
        functools.partial(sum_attr, attr="linear_attenuation_abs"),
    )
    add_feature(
        "cross_abs_final",
        ["linear_attenuation_abs"],
        functools.partial(nth_elem, attr="linear_attenuation_abs", n=-1),
    )
    add_feature(
        "cross_abs_mean",
        ["linear_attenuation_abs"],
        functools.partial(mean_attr, attr="linear_attenuation_abs"),
    )
    add_feature(
        "cross_abs_max",
        ["linear_attenuation_abs"],
        functools.partial(max_attr, attr="linear_attenuation_abs"),
    )
    add_feature(
        "cross_abs_ge_dist_sum",
        ["linear_attenuation_abs", "ge_distance_perm"],
        functools.partial(sum_mul_attrs, a1="linear_attenuation_abs", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_abs_ge_dist_final",
        ["linear_attenuation_abs", "ge_distance_perm"],
        functools.partial(last_mul_attrs, a1="linear_attenuation_abs", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_abs_ge_dist_mean",
        ["linear_attenuation_abs", "ge_distance_perm"],
        functools.partial(mean_mul_attrs, a1="linear_attenuation_abs", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_abs_ge_dist_max",
        ["linear_attenuation_abs", "ge_distance_perm"],
        functools.partial(max_mul_attrs, a1="linear_attenuation_abs", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_abs_dist_sum",
        ["linear_attenuation_abs", "distance_perm"],
        functools.partial(sum_mul_attrs, a1="linear_attenuation_abs", a2="distance_perm"),
    )
    add_feature(
        "cross_abs_dist_final",
        ["linear_attenuation_abs", "distance_perm"],
        functools.partial(last_mul_attrs, a1="linear_attenuation_abs", a2="distance_perm"),
    )
    add_feature(
        "cross_abs_dist_mean",
        ["linear_attenuation_abs", "distance_perm"],
        functools.partial(mean_mul_attrs, a1="linear_attenuation_abs", a2="distance_perm"),
    )
    add_feature(
        "cross_abs_dist_max",
        ["linear_attenuation_abs", "distance_perm"],
        functools.partial(max_mul_attrs, a1="linear_attenuation_abs", a2="distance_perm"),
    )
    add_feature(
        "cross_abs_min",
        ["linear_attenuation_abs"],
        functools.partial(min_attr, attr="linear_attenuation_abs"),
    )
    add_feature(
        "cross_abs_ge_dist_min",
        ["linear_attenuation_abs", "ge_distance_perm"],
        functools.partial(min_mul_attrs, a1="linear_attenuation_abs", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_abs_dist_min",
        ["linear_attenuation_abs", "distance_perm"],
        functools.partial(min_mul_attrs, a1="linear_attenuation_abs", a2="distance_perm"),
    )

    add_feature(
        "cross_compt_sum",
        ["linear_attenuation_compt"],
        functools.partial(sum_attr, attr="linear_attenuation_compt"),
    )
    add_feature(
        "cross_compt_mean",
        ["linear_attenuation_compt"],
        functools.partial(mean_attr, attr="linear_attenuation_compt"),
    )
    add_feature(
        "cross_compt_max",
        ["linear_attenuation_compt"],
        functools.partial(max_attr, attr="linear_attenuation_compt"),
    )
    add_feature(
        "cross_compt_ge_dist_sum",
        ["linear_attenuation_compt", "ge_distance_perm"],
        functools.partial(sum_mul_attrs, a1="linear_attenuation_compt", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_compt_ge_dist_mean",
        ["linear_attenuation_compt", "ge_distance_perm"],
        functools.partial(mean_mul_attrs, a1="linear_attenuation_compt", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_compt_ge_dist_max",
        ["linear_attenuation_compt", "ge_distance_perm"],
        functools.partial(max_mul_attrs, a1="linear_attenuation_compt", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_compt_dist_sum",
        ["linear_attenuation_compt", "distance_perm"],
        functools.partial(sum_mul_attrs, a1="linear_attenuation_compt", a2="distance_perm"),
    )
    add_feature(
        "cross_compt_dist_mean",
        ["linear_attenuation_compt", "distance_perm"],
        functools.partial(mean_mul_attrs, a1="linear_attenuation_compt", a2="distance_perm"),
    )
    add_feature(
        "cross_compt_dist_max",
        ["linear_attenuation_compt", "distance_perm"],
        functools.partial(max_mul_attrs, a1="linear_attenuation_compt", a2="distance_perm"),
    )
    add_feature(
        "cross_compt_min",
        ["linear_attenuation_compt"],
        functools.partial(min_attr, attr="linear_attenuation_compt"),
    )
    add_feature(
        "cross_compt_ge_dist_min",
        ["linear_attenuation_compt", "ge_distance_perm"],
        functools.partial(min_mul_attrs, a1="linear_attenuation_compt", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_compt_dist_min",
        ["linear_attenuation_compt", "distance_perm"],
        functools.partial(min_mul_attrs, a1="linear_attenuation_compt", a2="distance_perm"),
    )
    add_feature(
        "cross_compt_sum_nonfinal",
        ["linear_attenuation_compt"],
        functools.partial(sum_nonfinal, attr="linear_attenuation_compt"),
    )
    add_feature(
        "cross_compt_mean_nonfinal",
        ["linear_attenuation_compt"],
        functools.partial(mean_nonfinal, attr="linear_attenuation_compt"),
    )
    add_feature(
        "cross_compt_min_nonfinal",
        ["linear_attenuation_compt"],
        functools.partial(min_nonfinal, attr="linear_attenuation_compt"),
    )
    add_feature(
        "cross_compt_dist_sum_nonfinal",
        ["linear_attenuation_compt", "distance_perm"],
        functools.partial(sum_mul_attrs_nonfinal, a1="linear_attenuation_compt", a2="distance_perm"),
    )
    add_feature(
        "cross_compt_dist_mean_nonfinal",
        ["linear_attenuation_compt", "distance_perm"],
        functools.partial(mean_mul_attrs_nonfinal, a1="linear_attenuation_compt", a2="distance_perm"),
    )
    add_feature(
        "cross_compt_dist_min_nonfinal",
        ["linear_attenuation_compt", "distance_perm"],
        functools.partial(min_mul_attrs_nonfinal, a1="linear_attenuation_compt", a2="distance_perm"),
    )
    add_feature(
        "cross_compt_ge_dist_sum_nonfinal",
        ["linear_attenuation_compt", "ge_distance_perm"],
        functools.partial(sum_mul_attrs_nonfinal, a1="linear_attenuation_compt", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_compt_ge_dist_mean_nonfinal",
        ["linear_attenuation_compt", "ge_distance_perm"],
        functools.partial(mean_mul_attrs_nonfinal, a1="linear_attenuation_compt", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_compt_ge_dist_min_nonfinal",
        ["linear_attenuation_compt", "ge_distance_perm"],
        functools.partial(min_mul_attrs_nonfinal, a1="linear_attenuation_compt", a2="ge_distance_perm"),
    )


def _append_probability_features(add_feature, perm_calc):
    add_feature(
        "p_abs_sum",
        ["linear_attenuation_abs", "lin_mu_total"],
        lambda: njit_sum(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total),
    )
    add_feature(
        "p_abs_final",
        ["linear_attenuation_abs", "lin_mu_total"],
        lambda: perm_calc.linear_attenuation_abs[-1] / perm_calc.lin_mu_total[-1],
    )
    add_feature(
        "p_abs_mean",
        ["linear_attenuation_abs", "lin_mu_total"],
        lambda: njit_mean(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total),
    )
    add_feature(
        "p_abs_max",
        ["linear_attenuation_abs", "lin_mu_total"],
        lambda: njit_max(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total),
    )
    add_feature(
        "p_abs_min",
        ["linear_attenuation_abs", "lin_mu_total"],
        lambda: njit_min(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total),
    )
    add_feature(
        "-log_p_abs_sum",
        ["linear_attenuation_abs", "lin_mu_total"],
        lambda: njit_sum(-np.log(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total)),
    )
    add_feature(
        "-log_p_abs_final",
        ["linear_attenuation_abs", "lin_mu_total"],
        lambda: -np.log(perm_calc.linear_attenuation_abs[-1] / perm_calc.lin_mu_total[-1]),
    )
    add_feature(
        "-log_p_abs_mean",
        ["linear_attenuation_abs", "lin_mu_total"],
        lambda: njit_mean(-np.log(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total)),
    )
    add_feature(
        "-log_p_abs_max",
        ["linear_attenuation_abs", "lin_mu_total"],
        lambda: njit_max(-np.log(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total)),
    )
    add_feature(
        "-log_p_abs_min",
        ["linear_attenuation_abs", "lin_mu_total"],
        lambda: njit_min(-np.log(perm_calc.linear_attenuation_abs / perm_calc.lin_mu_total)),
    )
    add_feature(
        "p_compt_sum",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_sum(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total),
    )
    add_feature(
        "p_compt_mean",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_mean(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total),
    )
    add_feature(
        "p_compt_max",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_max(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total),
    )
    add_feature(
        "p_compt_min",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_min(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total),
    )
    add_feature(
        "p_compt_sum_nonfinal",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_sum(perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1]),
    )
    add_feature(
        "p_compt_mean_nonfinal",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_mean(perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1]),
    )
    add_feature(
        "p_compt_min_nonfinal",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_min(perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1]),
    )
    add_feature(
        "-log_p_compt_sum",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_sum(-np.log(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total)),
    )
    add_feature(
        "-log_p_compt_mean",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_mean(-np.log(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total)),
    )
    add_feature(
        "-log_p_compt_max",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_max(-np.log(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total)),
    )
    add_feature(
        "-log_p_compt_min",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_min(-np.log(perm_calc.linear_attenuation_compt / perm_calc.lin_mu_total)),
    )
    add_feature(
        "-log_p_compt_sum_nonfinal",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_sum(-np.log(perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1])),
    )
    add_feature(
        "-log_p_compt_mean_nonfinal",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_mean(-np.log(perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1])),
    )
    add_feature(
        "-log_p_compt_min_nonfinal",
        ["linear_attenuation_compt", "lin_mu_total"],
        lambda: njit_min(-np.log(perm_calc.linear_attenuation_compt[:-1] / perm_calc.lin_mu_total[:-1])),
    )


def _append_total_cross_features(add_feature, perm_calc):
    add_feature("cross_total_sum", ["lin_mu_total"], functools.partial(sum_attr, attr="lin_mu_total"))
    add_feature("cross_total_mean", ["lin_mu_total"], functools.partial(mean_attr, attr="lin_mu_total"))
    add_feature("cross_total_max", ["lin_mu_total"], functools.partial(max_attr, attr="lin_mu_total"))
    add_feature(
        "cross_total_ge_dist_sum",
        ["lin_mu_total", "ge_distance_perm"],
        functools.partial(sum_mul_attrs, a1="lin_mu_total", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_total_ge_dist_mean",
        ["lin_mu_total", "ge_distance_perm"],
        functools.partial(mean_mul_attrs, a1="lin_mu_total", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_total_ge_dist_max",
        ["lin_mu_total", "ge_distance_perm"],
        functools.partial(max_mul_attrs, a1="lin_mu_total", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_total_dist_sum",
        ["lin_mu_total", "distance_perm"],
        functools.partial(sum_mul_attrs, a1="lin_mu_total", a2="distance_perm"),
    )
    add_feature(
        "cross_total_dist_mean",
        ["lin_mu_total", "distance_perm"],
        functools.partial(mean_mul_attrs, a1="lin_mu_total", a2="distance_perm"),
    )
    add_feature(
        "cross_total_dist_max",
        ["lin_mu_total", "distance_perm"],
        functools.partial(max_mul_attrs, a1="lin_mu_total", a2="distance_perm"),
    )
    add_feature("cross_total_min", ["lin_mu_total"], functools.partial(min_attr, attr="lin_mu_total"))
    add_feature(
        "cross_total_ge_dist_min",
        ["lin_mu_total", "ge_distance_perm"],
        functools.partial(min_mul_attrs, a1="lin_mu_total", a2="ge_distance_perm"),
    )
    add_feature(
        "cross_total_dist_min",
        ["lin_mu_total", "distance_perm"],
        functools.partial(min_mul_attrs, a1="lin_mu_total", a2="distance_perm"),
    )


def _append_klein_nishina_features(add_feature, perm_calc):
    add_feature(
        "klein-nishina_rel_sum_sum",
        ["klein_nishina_relative_use_Ei"],
        lambda: njit_sum(perm_calc.klein_nishina_relative_use_Ei),
    )
    add_feature(
        "klein-nishina_rel_sum_mean",
        ["klein_nishina_relative_use_Ei"],
        lambda: njit_mean(perm_calc.klein_nishina_relative_use_Ei),
    )
    add_feature(
        "klein-nishina_rel_sum_max",
        ["klein_nishina_relative_use_Ei"],
        lambda: njit_max(perm_calc.klein_nishina_relative_use_Ei),
    )
    add_feature(
        "klein-nishina_rel_sum_min",
        ["klein_nishina_relative_use_Ei"],
        lambda: njit_min(perm_calc.klein_nishina_relative_use_Ei),
    )
    add_feature(
        "-log_klein-nishina_rel_sum_sum",
        ["klein_nishina_relative_use_Ei"],
        lambda: njit_sum(-np.log(perm_calc.klein_nishina_relative_use_Ei)),
    )
    add_feature(
        "-log_klein-nishina_rel_sum_mean",
        ["klein_nishina_relative_use_Ei"],
        lambda: njit_mean(-np.log(perm_calc.klein_nishina_relative_use_Ei)),
    )
    add_feature(
        "-log_klein-nishina_rel_sum_max",
        ["klein_nishina_relative_use_Ei"],
        lambda: njit_max(-np.log(perm_calc.klein_nishina_relative_use_Ei)),
    )
    add_feature(
        "-log_klein-nishina_rel_sum_min",
        ["klein_nishina_relative_use_Ei"],
        lambda: njit_min(-np.log(perm_calc.klein_nishina_relative_use_Ei)),
    )
    add_feature(
        "klein-nishina_rel_geo_sum",
        ["klein_nishina_relative"],
        lambda: njit_sum(perm_calc.klein_nishina_relative),
    )
    add_feature(
        "klein-nishina_rel_geo_mean",
        ["klein_nishina_relative"],
        lambda: njit_mean(perm_calc.klein_nishina_relative),
    )
    add_feature(
        "klein-nishina_rel_geo_max",
        ["klein_nishina_relative"],
        lambda: njit_max(perm_calc.klein_nishina_relative),
    )
    add_feature(
        "klein-nishina_rel_geo_min",
        ["klein_nishina_relative"],
        lambda: njit_min(perm_calc.klein_nishina_relative),
    )
    add_feature(
        "-log_klein-nishina_rel_geo_sum",
        ["klein_nishina_relative"],
        lambda: njit_sum(-np.log(perm_calc.klein_nishina_relative)),
    )
    add_feature(
        "-log_klein-nishina_rel_geo_mean",
        ["klein_nishina_relative"],
        lambda: njit_mean(-np.log(perm_calc.klein_nishina_relative)),
    )
    add_feature(
        "-log_klein-nishina_rel_geo_max",
        ["klein_nishina_relative"],
        lambda: njit_max(-np.log(perm_calc.klein_nishina_relative)),
    )
    add_feature(
        "-log_klein-nishina_rel_geo_min",
        ["klein_nishina_relative"],
        lambda: njit_min(-np.log(perm_calc.klein_nishina_relative)),
    )
    add_feature(
        "klein-nishina_sum_sum",
        ["klein_nishina_use_Ei"],
        lambda: njit_sum(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS),
    )
    add_feature(
        "klein-nishina_sum_mean",
        ["klein_nishina_use_Ei"],
        lambda: njit_mean(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS),
    )
    add_feature(
        "klein-nishina_sum_max",
        ["klein_nishina_use_Ei"],
        lambda: njit_max(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS),
    )
    add_feature(
        "klein-nishina_sum_min",
        ["klein_nishina_use_Ei"],
        lambda: njit_min(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS),
    )
    add_feature(
        "-log_klein-nishina_sum_sum",
        ["klein_nishina_use_Ei"],
        lambda: njit_sum(-np.log(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS)),
    )
    add_feature(
        "-log_klein-nishina_sum_mean",
        ["klein_nishina_use_Ei"],
        lambda: njit_mean(-np.log(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS)),
    )
    add_feature(
        "-log_klein-nishina_sum_max",
        ["klein_nishina_use_Ei"],
        lambda: njit_max(-np.log(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS)),
    )
    add_feature(
        "-log_klein-nishina_sum_min",
        ["klein_nishina_use_Ei"],
        lambda: njit_min(-np.log(perm_calc.klein_nishina_use_Ei * RANGE_PROCESS)),
    )
    add_feature(
        "klein-nishina_geo_sum",
        ["klein_nishina"],
        lambda: njit_sum(perm_calc.klein_nishina * RANGE_PROCESS),
    )
    add_feature(
        "klein-nishina_geo_mean",
        ["klein_nishina"],
        lambda: njit_mean(perm_calc.klein_nishina * RANGE_PROCESS),
    )
    add_feature(
        "klein-nishina_geo_max",
        ["klein_nishina"],
        lambda: njit_max(perm_calc.klein_nishina * RANGE_PROCESS),
    )
    add_feature(
        "klein-nishina_geo_min",
        ["klein_nishina"],
        lambda: njit_min(perm_calc.klein_nishina * RANGE_PROCESS),
    )
    add_feature(
        "-log_klein-nishina_geo_sum",
        ["klein_nishina"],
        lambda: njit_sum(-np.log(perm_calc.klein_nishina * RANGE_PROCESS)),
    )
    add_feature(
        "-log_klein-nishina_geo_mean",
        ["klein_nishina"],
        lambda: njit_mean(-np.log(perm_calc.klein_nishina * RANGE_PROCESS)),
    )
    add_feature(
        "-log_klein-nishina_geo_max",
        ["klein_nishina"],
        lambda: njit_max(-np.log(perm_calc.klein_nishina * RANGE_PROCESS)),
    )
    add_feature(
        "-log_klein-nishina_geo_min",
        ["klein_nishina"],
        lambda: njit_min(-np.log(perm_calc.klein_nishina * RANGE_PROCESS)),
    )


def _append_cluster_property_features(add_feature, perm_calc):
    add_feature("first_r", ["radii_perm"], functools.partial(nth_elem, attr="radii_perm", n=0))
    add_feature("final_r", ["radii_perm"], functools.partial(nth_elem, attr="radii_perm", n=-1))
    add_feature(
        "first_energy_ratio",
        ["energies_perm", "energy_sum"],
        functools.partial(first_div_by_scalar, arr_attr="energies_perm", scalar_attr="energy_sum"),
    )
    add_feature(
        "final_energy_ratio",
        ["energies_perm"],
        functools.partial(final_pair_ratio, arr_attr="energies_perm"),
    )
    add_feature(
        "first_is_not_largest",
        ["energies_perm"],
        functools.partial(any_greater_slice_first, attr="energies_perm"),
    )
    add_feature(
        "first_is_not_closest",
        ["radii_perm"],
        functools.partial(any_less_slice_first, attr="radii_perm"),
    )
    add_feature(
        "tango_variance",
        ["tango_estimates_perm"],
        functools.partial(var_attr, attr="tango_estimates_perm"),
    )
    add_feature(
        "tango_v_variance",
        ["tango_estimates_sigma_perm"],
        functools.partial(inv_sum_inv_sq, attr="tango_estimates_sigma_perm"),
    )
    add_feature("tango_sigma", ["tango_estimates_perm"], functools.partial(std_attr, attr="tango_estimates_perm"))
    add_feature(
        "tango_v_sigma",
        ["tango_estimates_sigma_perm"],
        functools.partial(lambda pc, attr, Nmi=None: np.sqrt(inv_sum_inv_sq(pc, attr)), attr="tango_estimates_sigma_perm"),
    )
    add_feature("escape_probability", ["escape_probability"], functools.partial(first_elem, attr="escape_probability"))
    add_feature(
        "-log_escape_probability",
        ["escape_probability"],
        functools.partial(neglog_scalar_attr, attr="escape_probability"),
    )


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
        if Nmi is None and permutation is not None and hasattr(permutation, "__len__"):
            Nmi = len(permutation)
        if permutation is not None and hasattr(permutation, "__len__") and len(permutation) == 1:
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
    specs = build_feature_specs(perm_calc=perm_calc, Nmi=Nmi)

    index = 0
    for spec in specs:
        if compute_mode:
            if boolean_vector[index]:
                if spec.compute_fn is None:
                    raise ValueError("Feature computation requested without a callable feature spec")
                features_vector[index] = spec.compute_fn()
        elif name_mode:
            names.append(spec.name)
        elif dependency_mode:
            dependencies_dict[spec.name] = list(spec.dependencies)
        index += 1

    if name_mode:
        return names
    if dependency_mode:
        return dependencies_dict
    return features_vector
