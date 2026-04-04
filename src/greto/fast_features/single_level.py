"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Features for single interactions
"""

from typing import Optional, Iterable
import numpy as np

import greto.physics as phys
from greto.fast_features.event_level import event_level_values
from greto.utils import njit_norm, EPS


def single_values(
    permutation: Iterable[int],
    event_calc: event_level_values,
    inner_radius: float = 23.5,
    outer_radius: float = 32.5,
    # start_point:int = 0,
    boolean_vector: Optional[Iterable[bool]] = None,
    name_mode: bool = False,
    dependency_mode: bool = False,
    all_computations: bool = False,
    number_of_values: int = 15,
):
    """
    Return all of the features for an individual cluster consisting of a single
    interaction

    We can consider these features as entirely separate from the other features
    for clusters, or we can allow them to overlap, or both. This is because the
    values that are computable for a single interaction are just as computable
    for every other kind of cluster. If we separate them, then we are
    acknowledging that singles require wholly different treatment that needs to
    be separately accounted for.

    What features can we actually generate with just the single (no other information available):
    - distance to target
    - linear attenuation coefficient
    - distance to outer shell
    - combinations of distance and attenuation
    - if we had more geometric information, we could get distance to nearest
      exit point of the active detector material
    """
    if isinstance(permutation, int):
        permutation = (permutation,)

    compute_mode = not name_mode and not dependency_mode

    if compute_mode:
        features_vector = np.zeros((number_of_values,))
        if len(permutation) > 1:
            return features_vector

    if compute_mode and boolean_vector is None:
        all_computations = True

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

    if compute_mode:
        # print(permutation)
        linear_attenuation = phys.lin_att_total_fit(
            event_calc.energy_matrix[permutation[0]]
        )
        # distance_to_inside = event_calc.ge_distance[start_point,permutation[0]]
        distance_to_inside = (
            njit_norm(event_calc.point_matrix[permutation[0]]) - inner_radius
        )
        if distance_to_inside < EPS:
            # print(f"Negative distance to the inside of the detector {distance_to_inside}, inner radius {inner_radius}")
            distance_to_inside = EPS
            # TODO - put a counter, log more information
            # raise ValueError
        distance_to_outside = outer_radius - inner_radius - distance_to_inside

    compute_value("penetration_cm", ["point_matrix"], lambda: distance_to_inside)
    index += 1

    compute_value("edge_cm", ["point_matrix"], lambda: distance_to_outside)
    index += 1

    compute_value(
        "linear_attenuation_cm-1", ["energy_matrix"], lambda: linear_attenuation
    )
    index += 1

    compute_value(
        "energy", ["energy_matrix"], lambda: event_calc.energy_matrix[permutation[0]]
    )
    index += 1

    compute_value(
        "pen_attenuation",
        ["point_matrix", "energy_matrix"],
        lambda: distance_to_inside * linear_attenuation,
    )
    index += 1

    compute_value(
        "pen_prob_remain",
        ["point_matrix", "energy_matrix"],
        lambda: np.exp(-distance_to_inside * linear_attenuation),
    )
    index += 1

    compute_value(
        "pen_prob_density",
        ["point_matrix", "energy_matrix"],
        lambda: linear_attenuation * np.exp(-distance_to_inside * linear_attenuation),
    )
    index += 1

    compute_value(
        "pen_prob_cumu",
        ["point_matrix", "energy_matrix"],
        lambda: 1 - np.exp(-distance_to_inside * linear_attenuation),
    )
    index += 1

    compute_value(
        "edge_attenuation",
        ["point_matrix", "energy_matrix"],
        lambda: distance_to_outside * linear_attenuation,
    )
    index += 1

    compute_value(
        "edge_prob_remain",
        ["point_matrix", "energy_matrix"],
        lambda: np.exp(-distance_to_outside * linear_attenuation),
    )
    index += 1

    compute_value(
        "edge_prob_density",
        ["point_matrix", "energy_matrix"],
        lambda: linear_attenuation * np.exp(-distance_to_outside * linear_attenuation),
    )
    index += 1

    compute_value(
        "edge_prob_cumu",
        ["point_matrix", "energy_matrix"],
        lambda: 1 - np.exp(-distance_to_outside * linear_attenuation),
    )
    index += 1

    compute_value("inv_pen", ["point_matrix"], lambda: 1.0 / distance_to_inside)
    index += 1

    compute_value("inv_edge", ["point_matrix"], lambda: 1.0 / distance_to_outside)
    index += 1

    compute_value(
        "interpolated_range",
        ["point_matrix", "energy_matrix"],
        lambda: phys.singles_depth_explicit(
            distance_to_inside, event_calc.energy_matrix[permutation[0]]
        ),
    )
    index += 1

    # return {
    #     "penetration_cm": distance_to_inside,
    #     "edge_cm": distance_to_outside,
    #     "linear_attenuation_cm-1": linear_attenuation,
    #     "energy": event.energy_matrix[permutation[0]],
    #     "pen_attenuation": distance_to_inside * linear_attenuation,
    #     "pen_prob_remain": np.exp(-distance_to_inside * linear_attenuation),
    #     "pen_prob_density": linear_attenuation * np.exp(-distance_to_inside * linear_attenuation),
    #     "pen_prob_cumu": 1 - np.exp(-distance_to_inside * linear_attenuation),
    #     "edge_attenuation": distance_to_outside * linear_attenuation,
    #     "edge_prob_remain": np.exp(-distance_to_outside * linear_attenuation),
    #     "edge_prob_density": linear_attenuation * np.exp(-distance_to_outside * linear_attenuation),
    #     "edge_prob_cumu": 1 - np.exp(-distance_to_outside * linear_attenuation),
    #     "inv_pen": 1.0 / distance_to_inside,
    #     "inv_edge": 1.0 / distance_to_outside,
    #     "interpolated_range": singles_depth(event, permutation, detector=detector),
    # }

    if name_mode:
        return names
    if dependency_mode:
        return dependencies_dict
    return features_vector
