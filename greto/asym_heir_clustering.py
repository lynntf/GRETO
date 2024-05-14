"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Asymmetric hierarchical clustering

Typical hierarchical clustering is done to combine nearby points obeying a
distance hierarchy. The distances between clusters at some level of the
hierarchy is symmetric, i.e., for clusters `A` and `B`, the distance `A -> B` is
the same as the distance from `B -> A`. In the case of gamma-ray interaction
sequences (or any causal sequence), the clusters `A` and `B` are ordered
sequences with an explicit order of head to tail and combining clusters `A -> B`
maintains that order (tail of `A` to head of `B`). This introduces asymmetry in
cluster distances (swapping the order exchanges heads and tails).

This ordered clustering is not typically desired as it results in long chains of
points rather than groups of points. However, long chains are the desired output
of gamma-ray tracking. In order to accommodate this, the hierarchical clustering
methods from `scipy` are extended here to handle the asymmetric distances that
this creates.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def combine_clusters(
    distances: np.ndarray[float], eligible: np.ndarray[bool] = None
) -> Tuple[int, int]:
    """
    # Find the pair of clusters with the minimum distance between them.

    This function takes a matrix of distances between clusters and an optional
    array of boolean values indicating which clusters are eligible to be joined.
    It returns the indices of the pair of clusters that have the smallest distance
    among all eligible pairs. The distance matrix can be either single, directional,
    or weighted directional.

    ## Args
    - `distances`: A square matrix of numbers representing the distances between
                      clusters. The shape of the matrix must be (n, n), where n is
                      the number of clusters. The diagonal elements are ignored.
    - `eligible` : An optional array of boolean values indicating which clusters
                     are eligible to be joined. The shape of the array must be (n,),
                     where n is the number of clusters. If `None`, all clusters are
                     considered eligible. The default value is `None`.
    ## Returns
        A tuple of two integers representing the indices of the pair of clusters
        that have the minimum distance between them. The indices are in the range
        [0, n), where n is the number of clusters.
    ## Raises
    - `ValueError`: If the distances matrix is not square or if the eligible array
                    does not match the shape of the distances matrix.
    """
    # Copy the distances matrix and mask out ineligible or diagonal elements
    A = np.copy(distances)
    if eligible is not None:
        if eligible.shape != A.shape:
            print(eligible.shape, A.shape)
            raise ValueError(
                "Eligible array does not match the shape of distances matrix."
            )
        A[~eligible] = np.max(distances)
    A[np.diag_indices(A.shape[0])] = np.inf

    # Find the indices of the minimum element in the masked matrix
    i, j = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
        np.argmin(A), A.shape  # pylint: disable=unbalanced-tuple-unpacking
    )

    # Return the indices as a tuple
    return i, j


def update_distances(
    distances: np.ndarray,
    i: int,
    j: int,
    method: str = "directional",
    weight: float = 0.1,
) -> Tuple[np.ndarray]:
    """
    # Update the distance matrix for clustering after joining two clusters.

    This function takes a matrix of distances between clusters, the indices of
    the two clusters that are joined, and the method of updating the distances.
    It returns a tuple of two arrays representing the updated row and column of
    the distance matrix for replacing the rows and columns of the two indices to join.

    ## Args
    - `distances`: A square matrix of numbers representing the distances between
      clusters. The shape of the matrix must be (n, n), where n is the number of
      clusters.
    - `i`: The index of the first cluster that is joined. It must be an integer
      in the range [0, n), where n is the number of clusters.
    - `j`: The index of the second cluster that is joined. It must be an integer
      in the range [0, n), where n is the number of clusters.
    - `method` : The method of updating the distances after joining two
      clusters.
    It can be one of the following values:
        - 'single': Use the minimum distance between the two clusters.
        - 'directional' (default): Use the minimum distance from the ends of the
          two clusters in a directional way (each cluster is head -> tail)
        - 'weighted_directional': Use a weighted average of the distances from
          both clusters to the other clusters.
    - `weight` : The weight used for calculating the weighted average distance in
    the 'weighted_directional' method. It must be a number between 0
    and 1. The default value is 0.1.
    ## Returns
    A tuple of two arrays representing the updated row and column of the
    distance matrix. The shape of each array is (n,), where n is the number of
    clusters.
    ## Raises
    - `ValueError` : If any of the parameters are invalid or incompatible with
    each other.
    """
    # Check if parameters are valid
    if not isinstance(distances, np.ndarray):
        raise ValueError("Distances must be a numpy array.")
    if distances.shape[0] != distances.shape[1]:
        raise ValueError("Distances must be a square matrix.")
    if i < 0 or i >= distances.shape[0] or j < 0 or j >= distances.shape[0]:
        raise ValueError(
            "Indices must be in range [0, n), where n is the number of clusters."
        )
    if not isinstance(method, str):
        raise ValueError("Method must be a string.")
    if method not in ["single", "directional", "weighted_directional"]:
        raise ValueError(
            "Method must be one of ['single', 'directional', 'weighted_directional']."
        )
    if not isinstance(weight, (int, float)):
        raise ValueError("Weight must be a number.")
    if weight < 0 or weight > 1:
        raise ValueError("Weight must be between 0 and 1.")

    # Update distances according to method
    if method == "single":
        update_row = np.minimum(distances[i, :], distances[j, :])
        update_column = np.minimum(distances[:, i], distances[:, j])
    elif method == "directional":
        update_row = distances[i, :]
        update_column = distances[:, j]
    elif method == "weighted_directional":
        update_row = (1 - weight) * distances[i, :] + weight * distances[j, :]
        update_column = weight * distances[:, i] + (1 - weight) * distances[:, j]

    # Return updated row and column as a tuple
    return update_row, update_column


def asym_hier_linkage(
    distances: np.ndarray,
    clusters: dict = None,
    max_cluster_length: int = None,
    debug: bool = False,
    method: str = "directional",
    weight: float = 0.5,
) -> np.ndarray:
    """
    # Perform asymmetric hierarchical clustering on a distance matrix.

    This module implements an asymmetric hierarchical clustering algorithm that
    accepts a maximum cluster size as a constraint. The algorithm starts with
    each element as a singleton cluster and iteratively joins the pair of
    clusters that have the minimum distance between them, until all elements are
    in one cluster or the maximum cluster size is reached (the clusters only
    grow past the maximum cluster size only once it is impossible to not do so).
    The distance matrix can be either single, directional, or weighted
    directional.

    ## Args
    - `distances` : A square matrix of numbers representing the distances
      between elements. The shape of the matrix must be (n, n), where n is the
      number of elements.
    - `clusters` : An optional dictionary of lists representing the initial
      clusters. The keys are cluster indices and the values are lists of element
      indices. If `None`, each element is assigned to its own cluster. The
      default value is `None`.
    - `max_cluster_size` : An optional integer representing the maximum size of
      a cluster. The algorithm will try to avoid joining clusters that exceed
      this size, unless it is impossible to do so. If `None`, there is no limit
      on the cluster size. The default value is `None`.
    - `debug` : An optional boolean indicating whether to print debugging messages.
      If `True`, the algorithm will print the selected clusters and their
      distances at each iteration. The default value is `False`.
    - `method` : An optional string indicating the method of updating the
      distances after joining two clusters. It can be one of the following
      values:
        - `'single'` : Use the minimum distance between the two clusters.
        - `'directional'` (default): Use the distance from the first cluster to
          the other clusters.
        - `'weighted_directional'` : Use a weighted average of the distances
          from both clusters to the other clusters.
    - `weight` : An optional float representing the weight used for calculating the
      weighted average distance in the 'weighted_directional' method. It
      must be a number between 0 and 1. The default value is 0.5.
    ## Returns
    A linkage array representing the hierarchical clustering result. The shape
    of the array is (n-1, 4), where n is the number of elements. Each row
    corresponds to one iteration of joining two clusters. The first and second
    columns are the indices of the joined clusters. The third column is the
    distance between them. The fourth column is the size of the resulting
    cluster.
    ## Raises
    - `ValueError` : If any of the parameters are invalid or incompatible with
      each other.
    """
    # Copy the distance matrix and initialize clusters if not given
    distances_copy = np.copy(distances)
    if clusters is None:
        clusters = [[i + 1] for i in range(distances_copy.shape[0])]
    N = len(clusters)
    if max_cluster_length is None:
        max_cluster_length = N

    # Create arrays for storing lengths, eligibility, validity, and indices of clusters
    lengths = np.ones((N,), dtype=int)
    combined_cluster_lengths = np.add.outer(lengths, lengths)
    eligible = np.array([len(cluster) <= max_cluster_length for cluster in clusters])
    valid = np.ones((N,), dtype=bool)
    inds = np.arange(N)

    # Create an array for storing linkage information and an array for shifting indices
    linkage = np.zeros((N - 1, 4))
    index_shift = np.arange(N)

    # Handle edge case when there is only one element
    if N == 1:
        linkage = np.zeros((1, 4))

    # Loop over N-1 iterations of joining two clusters
    for ii in range(N - 1):
        # Find the pair of eligible clusters with minimum distance between them
        if min(lengths[valid]) < max_cluster_length:
            eligible = combined_cluster_lengths <= max_cluster_length
            i, j = combine_clusters(
                (distances_copy[valid, :])[:, valid], (eligible[valid, :])[:, valid]
            )
        else:
            i, j = combine_clusters((distances_copy[valid, :])[:, valid])

        # Get the actual indices of the selected clusters from the valid indices array
        i = (inds[valid])[i]
        j = (inds[valid])[j]

        # Print debugging messages if debug flag is True
        if debug:
            print(
                f"Selected clusters: {distances_copy[i,j]:4.4f} {clusters[i]} and {clusters[j]}"
            )

        # Update lengths, linkage, index_shift, and clusters arrays
        lengths[min(i, j)] = lengths[i] + lengths[j]
        linkage[ii, 0] = index_shift[i]
        linkage[ii, 1] = index_shift[j]
        linkage[ii, 2] = distances_copy[i, j]
        linkage[ii, 3] = lengths[min(i, j)]
        index_shift[i] = N + ii
        index_shift[j] = N + ii
        combined_cluster_lengths[min(i, j), :] += lengths[max(i, j)]
        combined_cluster_lengths[:, min(i, j)] += lengths[max(i, j)]
        combined_cluster_lengths[min(i, j), min(i, j)] -= lengths[max(i, j)]
        clusters[min(i, j)] = clusters[i] + clusters[j]

        # Mark the maximum index of the selected clusters as invalid
        valid[max(i, j)] = False

        # Update the distance matrix using the update_distances function
        update_row, update_column = update_distances(
            distances_copy, i, j, method=method, weight=weight
        )
        distances_copy[min(i, j), :] = update_row
        distances_copy[:, min(i, j)] = update_column

    # Return the linkage array as the result
    return linkage
