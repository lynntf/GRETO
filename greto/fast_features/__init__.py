"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Attempt to make computing feature values not very slow
"""

from __future__ import annotations

from typing import Optional, List
from collections import namedtuple

import numpy as np

from greto.fast_features.event_level import event_level_values, event_values
from greto.fast_features.permutation_level import perm_level_values, perm_atoms
from greto.fast_features.feature_level import feature_values
from greto.fast_features.single_level import single_values
from greto.fast_features.cluster_level import cluster_feature_values, cluster_atoms
from greto.event_class import Event
from greto.utils import njit_any

event_value_names = event_values(None, None, None, None, name_mode=True)
number_of_event_values = len(event_value_names)
event_value_dependencies = event_values(None, None, None, None, dependency_mode=True)

perm_value_names = perm_atoms(None, None, name_mode=True)
number_of_perm_values = len(perm_value_names)
perm_value_dependencies = perm_atoms(None, None, dependency_mode=True)

feature_names = feature_values(None, None, name_mode=True)
number_of_feature_values = len(feature_names)
feature_dependencies = feature_values(None, None, dependency_mode=True)

feature_names_tango = [name + "_tango" for name in feature_names]

single_feature_names = single_values(None, None, name_mode=True)
number_of_single_feature_values = len(single_feature_names)
single_feature_dependencies = single_values(None, None, dependency_mode=True)

cluster_calc_feature_names = cluster_atoms(None, None, name_mode=True)
number_of_cluster_calc_feature_values = len(cluster_calc_feature_names)
cluster_calc_feature_dependencies = cluster_atoms(None, None, dependency_mode=True)

cluster_feature_names = cluster_feature_values(None, name_mode=True)
number_of_cluster_feature_values = len(cluster_feature_names)
cluster_feature_dependencies = cluster_feature_values(None, dependency_mode=True)


boolean_vectors = namedtuple('boolean_vectors', [
    "event_boolean_vector",
    "perm_boolean_vector",
    "feature_boolean_vector",
    "perm_boolean_vector_tango",
    "feature_boolean_vector_tango",
    "single_feature_boolean_vector",
    "cluster_calc_feature_boolean_vector",
    "cluster_feature_boolean_vector",
])


def convert_feature_names_to_boolean_vectors(list_of_value_names: List[str])->boolean_vectors:
    """
    Given some list of values that one would like to compute (at any level), we
    can convert that list of values to boolean vectors that we can then pass to
    the calculators to compute just those values

    Args:
        - list_of_value_names: the names of the values that one would like
          (feature values, permutation level values, or event level values)

    Returns:
        - event_boolean_vector: booleans for event level computations
        - perm_boolean_vector: booleans for permutation level computations
        - perm_tango_boolean_vector: booleans for permutation level computations
        - feature_boolean_vector: booleans for feature level computations
        - feature_tango_boolean_vector: booleans for feature level computations
        - single_feature_boolean_vector: booleans for single feature level
          computations
        - cluster_calc_feature_boolean_vector: booleans for cluster level
          calculations (order independent)
        - cluster_feature_boolean_vector: booleans for features from cluster
          level
    """

    _event_value_names = event_values(None, None, None, None, name_mode=True)
    _event_value_dependencies = event_values(None, None, None, None, dependency_mode=True)

    _perm_value_names = perm_atoms(None, None, name_mode=True)
    _perm_value_dependencies = perm_atoms(None, None, dependency_mode=True)

    _feature_names = feature_values(None, None, name_mode=True)
    _feature_dependencies = feature_values(None, None, dependency_mode=True)

    tango_mode_extra_features = ["estimate_start_energy_sigma_weighted_perm"]

    _single_feature_names = single_values(None, None, name_mode=True)
    _single_feature_dependencies = single_values(None, None, dependency_mode=True)

    _cluster_calc_feature_names = cluster_atoms(None, None, name_mode=True)
    _cluster_calc_feature_dependencies = cluster_atoms(None, None, dependency_mode=True)

    _cluster_feature_names = cluster_feature_values(None, name_mode=True)
    _cluster_feature_dependencies = cluster_feature_values(None, dependency_mode=True)

    # Split the tango and non-tango features
    list_of_nontango_value_names = [name for name in list_of_value_names if not name.endswith("_tango")]
    list_of_tango_value_names = [name[:-6] for name in list_of_value_names if name.endswith("_tango")]

    # Deal with the non-tango features first
    total_dependencies = set(list_of_nontango_value_names)
    old_num_dependencies = 0
    num_dependencies = len(total_dependencies)
    while old_num_dependencies < num_dependencies:
        old_num_dependencies = num_dependencies
        new_dependencies = set()
        for name in total_dependencies:
            for new_name in _feature_dependencies.get(name, []):
                new_dependencies.add(new_name)
            for new_name in _perm_value_dependencies.get(name, []):
                new_dependencies.add(new_name)
            for new_name in _event_value_dependencies.get(name, []):
                new_dependencies.add(new_name)
            for new_name in _single_feature_dependencies.get(name, []):
                new_dependencies.add(new_name)
            for new_name in _cluster_calc_feature_dependencies.get(name, []):
                new_dependencies.add(new_name)
            for new_name in _cluster_feature_dependencies.get(name, []):
                new_dependencies.add(new_name)
        total_dependencies = total_dependencies | new_dependencies
        num_dependencies = len(total_dependencies)

    event_boolean_vector = []
    for name in _event_value_names:
        if name in total_dependencies:
            event_boolean_vector.append(True)
        else:
            event_boolean_vector.append(False)

    perm_boolean_vector = []
    for name in _perm_value_names:
        if name in total_dependencies:
            perm_boolean_vector.append(True)
        else:
            perm_boolean_vector.append(False)

    feature_boolean_vector = []
    for name in _feature_names:
        if name in total_dependencies:
            feature_boolean_vector.append(True)
        else:
            feature_boolean_vector.append(False)

    single_feature_boolean_vector = []
    for name in _single_feature_names:
        if name in total_dependencies:
            single_feature_boolean_vector.append(True)
        else:
            single_feature_boolean_vector.append(False)

    cluster_calc_feature_boolean_vector = []
    for name in _cluster_calc_feature_names:
        if name in total_dependencies:
            cluster_calc_feature_boolean_vector.append(True)
        else:
            cluster_calc_feature_boolean_vector.append(False)

    cluster_feature_boolean_vector = []
    for name in _cluster_feature_names:
        if name in total_dependencies:
            cluster_feature_boolean_vector.append(True)
        else:
            cluster_feature_boolean_vector.append(False)

    # Deal with the tango values separately
    if len(list_of_tango_value_names) > 0: # if there are any of these features at all
        total_dependencies = set(list_of_tango_value_names + tango_mode_extra_features)
        old_num_dependencies = 0
        num_dependencies = len(total_dependencies)
        while old_num_dependencies < num_dependencies:
            old_num_dependencies = num_dependencies
            new_dependencies = set()
            for name in total_dependencies:
                for new_name in _feature_dependencies.get(name, []):
                    new_dependencies.add(new_name)
                for new_name in _perm_value_dependencies.get(name, []):
                    new_dependencies.add(new_name)
                for new_name in _event_value_dependencies.get(name, []):
                    new_dependencies.add(new_name)
            total_dependencies = total_dependencies | new_dependencies
            num_dependencies = len(total_dependencies)
    else:
        total_dependencies = set()

    event_boolean_vector_tango = []
    for name in _event_value_names:
        if name in total_dependencies:
            event_boolean_vector_tango.append(True)
        else:
            event_boolean_vector_tango.append(False)

    perm_boolean_vector_tango = []
    for name in _perm_value_names:
        if name in total_dependencies:
            perm_boolean_vector_tango.append(True)
        else:
            perm_boolean_vector_tango.append(False)

    feature_boolean_vector_tango = []
    for name in _feature_names:
        if name in total_dependencies:
            feature_boolean_vector_tango.append(True)
        else:
            feature_boolean_vector_tango.append(False)

    return boolean_vectors(
        np.logical_or(np.array(event_boolean_vector), np.array(event_boolean_vector_tango)),
        np.array(perm_boolean_vector),
        np.array(feature_boolean_vector),
        np.array(perm_boolean_vector_tango),
        np.array(feature_boolean_vector_tango),
        np.array(single_feature_boolean_vector),
        np.array(cluster_calc_feature_boolean_vector),
        np.array(cluster_feature_boolean_vector),
    )


def get_event_level_values(
    event: Event, event_level_boolean_vector: Optional[np.ndarray], eres: float = 1e-3
) -> event_level_values:
    """Pass event values to the event_values function to get the computed values"""
    return event_values(
        event.point_matrix,
        event.energy_matrix,
        event.position_uncertainty,
        event.detector_config.inner_radius,
        number_of_event_values,
        False,
        False,
        False,
        event_level_boolean_vector,
        eres,
    )


def get_perm_features(
    event: Event,
    event_calc: event_level_values,
    permutation: tuple[int],
    start_point: int = 0,
    start_energy: float = None,
    Nmi: int = None,
    perm_boolean_vector: np.ndarray = None,
    feature_boolean_vector: np.ndarray = None,
    perm_tango_boolean_vector: np.ndarray = None,
    feature_tango_boolean_vector: np.ndarray = None,
    eres: float = 1e-3,
):
    """Given an event and the corresponding event calculations, get the
    associated features"""
    if Nmi is None:
        Nmi = len(permutation)

    if perm_boolean_vector is not None and njit_any(perm_boolean_vector):
        perm_values = perm_atoms(
            permutation,
            event_calc,
            start_point,
            start_energy,
            True,
            Nmi,
            2 * np.pi,
            event.detector_config.outer_radius,
            number_of_perm_values,
            False,
            False,
            False,
            False,
            perm_boolean_vector,
            eres,
        )

        features = feature_values(
            permutation,
            perm_values,
            Nmi,
            feature_boolean_vector,
            False,
            False,
            False,
            number_of_feature_values,
        )
    else:
        features = np.zeros((number_of_feature_values,))

    if perm_tango_boolean_vector is not None and njit_any(perm_tango_boolean_vector):
        perm_values_tango = perm_atoms(
            permutation,
            event_calc,
            start_point,
            start_energy,
            True,
            Nmi,
            2 * np.pi,
            event.detector_config.outer_radius,
            number_of_perm_values,
            False,
            False,
            True,
            False,
            perm_tango_boolean_vector,
            eres,
        )

        tango_features = feature_values(
            permutation,
            perm_values_tango,
            Nmi,
            feature_tango_boolean_vector,
            False,
            False,
            False,
            number_of_feature_values,
        )
    else:
        tango_features = np.zeros(features.shape)

    return features, tango_features


def get_cluster_features(
    event_calc: event_level_values,
    permutation: tuple[int],
    Nmi: int = None,
    cluster_calc_boolean_vector: np.ndarray = None,
    cluster_boolean_vector: np.ndarray = None,
):
    """Given an event and the corresponding event calculations, get the
    associated features"""
    if Nmi is None:
        Nmi = len(permutation)

    cluster_calc = cluster_atoms(
        permutation,
        event_calc,
        number_of_cluster_calc_feature_values,
        False,
        False,
        False,
        cluster_calc_boolean_vector,
    )

    return cluster_feature_values(
        cluster_calc,
        number_of_cluster_feature_values,
        False,
        False,
        False,
        cluster_boolean_vector,
    )


def get_single_features(
    event: Event,
    event_calc: event_level_values,
    permutation: tuple[int],
    singles_boolean_vector: np.ndarray = None,
    # start_point:int = 0,
):
    """Given an event and the corresponding event calculations, get the
    associated features"""
    return single_values(
        permutation,
        event_calc,
        event.detector_config.inner_radius,
        event.detector_config.outer_radius,
        # start_point,
        singles_boolean_vector,
        False,
        False,
        False,
        number_of_single_feature_values,
    )
