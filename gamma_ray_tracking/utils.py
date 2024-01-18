"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Utility functions
"""
from copy import deepcopy
from functools import lru_cache
from typing import BinaryIO, Dict, Iterable, Tuple

import numpy as np
from scipy.interpolate import interp1d


@lru_cache(maxsize=100)
def perm_to_transition(permutation: Tuple, D: int = 3) -> Tuple[np.ndarray]:
    """
    Transform a permutation to transition indices with dimension D

    Returns D arrays for transitions from out[0] -> out[1] -> ... -> out[D]

    We apply caching to take advantage of multiple calls to this function during
    a single FOM evaluation.
    """
    p = np.array(permutation)
    return tuple(p[i : len(p) - (D - i) + 1] for i in range(D))


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
    for i, partitions in enumerate(previous_partitions):
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
