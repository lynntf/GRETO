"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Event level values
"""

from __future__ import annotations

from collections import namedtuple
from typing import Dict, List, Tuple, Union

import numpy as np

import greto.geometry as geo
import greto.physics as phys

event_level_values = namedtuple(
    "event_level_values",
    [
        "point_matrix",
        "energy_matrix",
        "distance",
        # 'angle_distance',
        "ge_distance",
        "cos_act",
        "cos_err",
        # 'theta_err',
        "tango_estimates",
        "tango_partial_derivatives",
        "tango_estimates_sigma",
        "radii",
    ],
)


def event_values(
    point_matrix: np.ndarray,
    energy_matrix: np.ndarray,
    position_uncertainty: np.ndarray,
    inner_radius: float,
    number_of_values: int = 8,
    name_mode: bool = False,
    dependency_mode: bool = False,
    all_computations: bool = False,
    boolean_vector: np.ndarray = None,
    eres: float = 1e-3,
) -> Union[Tuple[np.ndarray, ...], List[str], Dict[str, List[str]]]:
    """
    Event level computations

    Given an event's information, get the relevant values

    `name_mode` produces the names of the computed values referenced in
    dependencies for the `permutation_level` calculations

    Args:
        - point_matrix: event matrix of interaction point locations
        - energy_matrix: event matrix of interaction point energies
        - position_uncertainty: event matrix of interaction location uncertainty
        - inner_radius: detector inner radius [cm]
        - number_of_values: size of the boolean_vector (can compute this using
          the len() of the name_mode output)
        - name_mode: return the names of computed objects
        - dependency_mode: return the dependencies of computed objects (on the
          current computational level)
        - all_computations: compute all values
        - boolean_vector: vector indicating which values to compute
        - eres: energy resolution

    Returns:
        - distance: Euclidean distance between interaction points
        - ge_distance: distance through detector active material between
          interaction points
        - cos_act: actual cosines between scatters for interactions i->j->k
          (k->j->i)
        - tango_estimates: incoming energy estimate for interactions i->j->k
          (k->j->i)
        - tango_partial_derivatives: derivatives with respect to [0] energy
          (d_de) and [1] cosine theta (d_d_cos) for interactions i->j->k
        - tango_estimates_sigma: standard error in tango estimate for i->j->k
        - names: computational names (for dependencies) if name_mode
        - dependencies_dict: dictionary of dependencies if dependency_mode
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
                value = compute_fn()
            else:
                value = None
            return value
        elif name_mode:
            names.append(name)
        elif dependency_mode:
            dependencies_dict[name] = dependencies

    names = []
    dependencies_dict = {}
    index = 0

    distance = compute_value(
        "distance", [], lambda: geo.njit_square_pdist(point_matrix)
    )
    index += 1

    # angle_distance = compute_value(
    #     "angle_distance", [],
    #     lambda: np.arccos(1.0 - geo.njit_square_cosine_pdist(point_matrix[1:]))
    # )
    # index += 1

    ge_distance = compute_value(
        "ge_distance",
        ["distance"],
        lambda: geo.njit_square_ge_pdist(
            point_matrix,
            inner_radius=inner_radius,
            d12_euc=distance,
        ),
    )
    index += 1

    cos_act = compute_value("cos_act", [], lambda: geo.cosine_ijk(point_matrix))
    index += 1

    cos_err = compute_value(
        "cos_err",
        ["distance", "cos_act"],
        lambda: geo.err_cos_vec_precalc(distance, cos_act, position_uncertainty),
    )
    index += 1

    # theta_err = compute_value(
    #     "theta_err", ["distance", "cos_act"],
    #     lambda: geo.err_theta_vec_precalc(distance, cos_act, position_uncertainty)
    # )
    # index += 1

    tango_estimates = compute_value(
        "tango_estimates",
        ["cos_act"],
        lambda: phys.tango_incoming_estimate(energy_matrix, 1 - cos_act),
    )
    index += 1

    tango_partial_derivatives = compute_value(
        "tango_partial_derivatives",
        ["cos_act"],
        lambda: phys.partial_tango_incoming_derivatives(energy_matrix, 1 - cos_act),
    )
    index += 1

    tango_estimates_sigma = compute_value(
        "tango_estimates_sigma",
        ["tango_partial_derivatives", "cos_err"],
        lambda: np.sqrt(
            (eres * tango_partial_derivatives[0]) ** 2
            + (cos_err * tango_partial_derivatives[1]) ** 2
        ),
    )
    index += 1

    radii = compute_value("radii", [], lambda: geo.radii(point_matrix))
    index += 1

    if name_mode:
        return names
    if dependency_mode:
        return dependencies_dict
    return event_level_values(
        point_matrix,
        energy_matrix,
        distance,
        # angle_distance,
        ge_distance,
        cos_act,
        cos_err,
        # theta_err,
        tango_estimates,
        tango_partial_derivatives,
        tango_estimates_sigma,
        radii,
    )


# %%
# fmt: off
# def event_values_explicit(
#     point_matrix:np.ndarray,
#     energy_matrix:np.ndarray,
#     position_uncertainty:np.ndarray,
#     inner_radius:float,
#     name_mode:bool = False,
#     dependency_mode:bool = False,
#     all_computations:bool = False,
#     boolean_vector:np.ndarray = None,
#     eres = 1e-3
#     ):
#     """
#     Event level computations
#     """
#     compute_mode = not name_mode and not dependency_mode
#     if compute_mode and boolean_vector is None:
#         all_computations = True

#     distance = geo.njit_square_pdist(point_matrix)
#     angle_distance = np.arccos(1.0 - geo.njit_square_cosine_pdist(point_matrix[1:]))
#     ge_distance = geo.njit_square_ge_pdist(point_matrix,inner_radius=inner_radius, d12_euc=distance,)
#     cos_act = geo.cosine_ijk(point_matrix)
#     cos_err = geo.err_cos_vec_precalc(distance,cos_act,position_uncertainty,)
#     theta_err = geo.err_theta_vec_precalc(distance,cos_act,position_uncertainty,)
#     tango_estimates = phys.tango_incoming_estimate(energy_matrix,1 - cos_act,)
#     tango_partial_derivatives = phys.partial_tango_incoming_derivatives(energy_matrix,1 - cos_act,)
#     tango_estimates_sigma = np.sqrt((eres * tango_partial_derivatives[0]) ** 2+ (cos_err* tango_partial_derivatives[1])** 2)

# fmt: on
