"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Permutation level values
"""
from __future__ import annotations

from collections import namedtuple

import numba
import numpy as np

import greto.geometry as geo
from greto.utils import njit_norm, njit_mean
from greto.fast_features.event_level import event_level_values

cluster_level_values = namedtuple('cluster_level_values', [
    "n",
    # "centroid",
    "centroid_r",
    "average_r",
    # "length_width",
    "length",
    "width",
    "aspect_ratio",
])


@numba.njit
def length_width_func(point_matrix, transition_1D, centroid):
    """Basically does PCA on cluster, getting the eigenvalues of the covariance
    matrix"""
    return np.sqrt(np.linalg.eigvalsh(np.cov(point_matrix[transition_1D] - centroid, rowvar=False)))

def cluster_atoms(
    cluster:tuple[int],
    event_calc: event_level_values,
    number_of_values:int = 8,
    name_mode:bool = False,
    dependency_mode: bool = False,
    all_computations: bool = False,
    boolean_vector: np.ndarray = None,
):
    """
    Inputs are general computational atoms
    
    Outputs are cluster computational atoms (order doesn't matter)
    """
    compute_mode = not name_mode and not dependency_mode
    if compute_mode and boolean_vector is None:
        all_computations = True
    if all_computations:
        boolean_vector = np.ones((number_of_values,), dtype=np.bool_)


    def compute_value(name, dependencies, compute_fn):
        """
        Computes the value of the compute_fn or the name or dependencies
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

    if compute_mode:
        transition_1D = np.array(cluster)

    n = compute_value(
        "n", [],
        lambda: len(cluster)
    )
    index += 1

    centroid = compute_value(
        "centroid", ["point_matrix"],
        lambda: geo.centroid(event_calc.point_matrix[transition_1D])
    )
    index += 1

    centroid_r = compute_value(
        "centroid_r", ["centroid"],
        lambda: njit_norm(centroid)
    )
    index += 1

    average_r = compute_value(
        "average_r", ["radii"],
        lambda: njit_mean(event_calc.radii[transition_1D[1:]])
    )
    index += 1

    length_width = compute_value(
        "length_width", ["point_matrix", "centroid"],
        lambda: length_width_func(event_calc.point_matrix, transition_1D, centroid)
    )
    index += 1

    length = compute_value(
        "length", ["length_width"],
        lambda: length_width[-1]
    )
    index += 1

    width = compute_value(
        "width", ["length_width"],
        lambda: np.sqrt(length_width[0] * length_width[1])
    )
    index += 1

    aspect_ratio = compute_value(
        "aspect_ratio", ["length", "width"],
        lambda: length/width
    )
    index += 1

    if name_mode:
        return names
    if dependency_mode:
        return dependencies_dict
    return cluster_level_values(
        n,
        # centroid,
        centroid_r,
        average_r,
        # length_width,
        length,
        width,
        aspect_ratio,
    )


def cluster_feature_values(
    cluster_calc: cluster_level_values,
    number_of_values:int = 6,
    name_mode:bool = False,
    dependency_mode: bool = False,
    all_computations: bool = False,
    boolean_vector: np.ndarray = None,
):
    """Given calculated values, return them"""
    compute_mode = not name_mode and not dependency_mode
    if compute_mode:
        features_vector = np.ones((number_of_values,))
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
                features_vector[index] = compute_fn()
        elif name_mode:
            names.append(name)
        elif dependency_mode:
            dependencies_dict[name] = dependencies

    names = []
    dependencies_dict = {}
    index = 0

    compute_value("n", ["n"], lambda: cluster_calc.n)
    index += 1

    compute_value("centroid_r", ["centroid_r"], lambda: cluster_calc.centroid_r)
    index += 1

    compute_value("average_r", ["average_r"], lambda: cluster_calc.average_r)
    index += 1

    compute_value("length", ["length"], lambda: cluster_calc.length)
    index += 1

    compute_value("width", ["width"], lambda: cluster_calc.width)
    index += 1

    compute_value("aspect_ratio", ["aspect_ratio"], lambda: cluster_calc.aspect_ratio)
    index += 1

    if name_mode:
        return names
    if dependency_mode:
        return dependencies_dict
    return features_vector
