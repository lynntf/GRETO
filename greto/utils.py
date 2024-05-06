"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Utility functions
"""

from copy import deepcopy
from functools import cache, lru_cache
from itertools import permutations
from typing import BinaryIO, Dict, Generator, Iterable, Tuple

import numba
import numpy as np
from scipy.interpolate import interp1d


@lru_cache(maxsize=100)
def perm_to_transition(
    permutation: Tuple | np.ndarray, D: int = 3
) -> Tuple[np.ndarray]:
    """
    Transform a permutation to transition indices with dimension D

    Returns D arrays for transitions from out[0] -> out[1] -> ... -> out[D]

    We apply caching to take advantage of multiple calls to this function during
    a single FOM evaluation.

    >>> permutation = (1, 2, 3, 4, 5, 6, 7, 8, 9)  # A permutation
    >>> A = np.arange(1000).reshape((10, 10, 10))  # A 3D array
    >>> A[perm_to_transition(permutation, D = 3)]  # Select i->j->k from permutation
    array([123, 234, 345, 456, 567, 678])
    """
    if isinstance(permutation, tuple):
        permutation = np.array(permutation)
    return tuple(permutation[i : len(permutation) - (D - i) + 1] for i in range(D))


def perm_to_transition_2D(permutation: Tuple | np.ndarray) -> Tuple[np.ndarray]:
    """
    Transform a permutation to transition indices with dimension 2

    Returns 2 arrays for transitions from out[0] -> out[1]
    """
    if isinstance(permutation, tuple):
        permutation = np.array(permutation)
    return (
        permutation[0 : len(permutation) - 1],
        permutation[1 : len(permutation) - 0],
    )


def perm_to_transition_3D(permutation: Tuple | np.ndarray) -> Tuple[np.ndarray]:
    """
    Transform a permutation to transition indices with dimension 3

    Returns 3 arrays for transitions from out[0] -> out[1] -> out[2]
    """
    if isinstance(permutation, tuple):
        permutation = np.array(permutation)
    return (
        permutation[0 : len(permutation) - 2],
        permutation[1 : len(permutation) - 1],
        permutation[2 : len(permutation) - 0],
    )


@numba.njit
def reverse_cumsum(x: np.ndarray) -> np.ndarray:
    """
    Reversed cumulative sum for the vector x

    Args:
        - x: array we want the reverse cumulative sum of

    Returns:
        - reversed cumulative sum
    """
    return np.cumsum(x[::-1])[::-1]  # Reversed cumulative sum
    # return np.flip(np.flip(x, 0).cumsum(), 0)  # slightly slower for short vec
    # out = np.zeros((x.shape[0],))
    # out[-1] = x[-1]
    # for i in range(x.shape[0]-1, 0, -1):
    #     out[i-1] = out[i] + x[i-1]
    # return out

@numba.njit
def cumsum(x:np.ndarray) -> np.ndarray:
    """Compiled cumulative sum"""
    return np.cumsum(x)


@numba.njit
def njit_norm(a):
    """Vector norm"""
    b = np.sum(a**2)
    return np.sqrt(b)


@numba.njit
def njit_squared_norm(a):
    """Squared norm of vectors"""
    return np.sum(a**2)


@numba.njit
def njit_sum(a):
    """sum of a vector"""
    return np.sum(a)


@numba.njit
def njit_mean(a):
    """mean of a vector"""
    return np.sum(a) / len(a)

@numba.njit
def njit_any(a):
    """determine if any of the values are True"""
    for i in numba.prange(len(a)):
        if a[i]:
            return True
    return False

@numba.njit
def njit_min(a):
    """min of a vector"""
    return np.min(a)

@numba.njit
def njit_max(a):
    """min of a vector"""
    return np.max(a)


def perm_cumulative_sums(
    values: Iterable, r: int = None
) -> Generator[Tuple[Tuple[int], Tuple, np.ndarray, np.ndarray], None, None]:
    """
    A generator for cumulative and reverse cumulative sums with caching

    Because of redundancies in the values that can exist for cumulative sums of
    permuted values, there is some slight efficiency gains that we can get by
    simply caching some of the sum values. As an example, consider that the
    total sum of N numbers is the same regardless of their permutation.
    Likewise, there are only N possible values for sums of 1 value (and N-1
    values). Because of how many permutations there are, this may not be the
    most efficient method for larger N and the space the cache takes up can blow
    up. Because 5! = 120 ~ 5^3 = 125, we switch to an lru_cache for larger N
    (this can still result in savings sometimes)

    Args:
        - values: values to get the cumulative and reverse cumulative sums of
        - r: width of the permutation (get all permutations of width r)

    Returns:
        - index_perm: permutation of indices associated with the values
        - values_perm: the values permuted according to the index_perm
        - cumsum: cumulative sum of the permuted values
        - rev_cumsum: reverse cumulative sum of the permuted values
    """
    if len(values < 5):

        @cache
        def cache_sum(sorted_values):
            """Compute the cumulative sum for a given set of sorted values."""
            return sum(sorted_values)

    else:

        @lru_cache(maxsize=5**3)
        def cache_sum(sorted_values):
            """Compute the cumulative sum for a given set of sorted values."""
            return sum(sorted_values)

    for index_perm, values_perm in zip(
        permutations(range(len(values)), r=r), permutations(values, r=r)
    ):
        cumsum = []  # Initialize with 0 for the empty prefix
        rev_cumsum = []
        for i in range(len(values_perm)):
            prefix = cache_sum(tuple(sorted(values_perm[: i + 1])))
            suffix = cache_sum(tuple(sorted(values_perm[i:])))
            cumsum.append(prefix)
            rev_cumsum.append(suffix)
        yield index_perm, values_perm, np.array(cumsum), np.array(rev_cumsum)


def log_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Logarithmic interpolation with logarithmic extrapolation.
    """
    l_interp = interp1d(
        np.log(xp), np.log(fp), kind="slinear", fill_value="extrapolate"
    )
    return np.exp(l_interp(np.log(x)))


def get_file_size(file: BinaryIO) -> int:
    """Get the file size (in bytes) of an opened file object"""
    # Get the current position of the read cursor
    current_position = file.tell()

    # Move the cursor to the end of the file
    file.seek(0, 2)  # 2 means seeking relative to the end of the file

    # Get the size of the file
    file_size = file.tell()

    # Move the cursor back to the original position
    file.seek(current_position)

    return file_size


@lru_cache(maxsize=10)
def get_ordered_partitions(items: Iterable, max_items: int = 9) -> Dict[int, Tuple]:
    """
    Get all possible ordered partitions of the items.

    Take the items supplied and return all possible ordered partitions of the
    items. The partitions of the data describe all of the unique ways that the
    items can be separated from one another. This function returns every
    possible permutation of each partition such that all possible ordered
    partition is included in the list.

    The number of ordered partitions returned follows OEIS A000262
    (https://oeis.org/A000262), a sequence describing the number of "sets of
    lists". The number of ordered partitions grows very quickly (4 596 553
    partitions for 9 items) so a limit is added to the function to avoid
    accidental computer freezing computation.

    Parameters:
        `items` (`Iterable`): The items to be partitioned.

    Returns:
        `Dict[int, Tuple]`: A dictionary where keys represent the partition number and
                          values are tuples containing the ordered partitions.
    """
    if len(items) >= max_items:
        raise ValueError("Too many items requested.")
    # Base case: if there's only one item, return it as a single partition
    if len(items) <= 1:
        return [{1: tuple(items)}]

    # Get ordered partitions for the items excluding the last element
    previous_partitions = get_ordered_partitions(items[:-1])

    # Last element of the items list
    last_element = items[-1]

    # Initialize the result list to store the updated partitions
    updated_partitions = []

    # Iterate through each partition from the previous step
    for _, partitions in enumerate(previous_partitions):
        # Iterate through each position to insert the last element
        for j, partition in partitions.items():
            for k in range(len(partition) + 1):
                # Create a new partition by inserting the last element at the given position
                new_partitions = deepcopy(partitions)
                new_partitions[j] = tuple(
                    list(new_partitions[j][:k])
                    + [last_element]
                    + list(new_partitions[j][k:])
                )
                updated_partitions.append(new_partitions)

        # Create a new partition by appending the last element as a new singleton partition
        updated_partitions.append(partitions | {len(partitions) + 1: (last_element,)})

    return updated_partitions
