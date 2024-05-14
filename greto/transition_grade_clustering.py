"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Methods for generating features for transition grade clustering
"""

from typing import Dict, Iterable, List

import numpy as np

# from greto import default_config
# from greto.detector_config_class import DetectorConfig
from greto.event_class import Event

# from greto.interaction_class import Interaction
from greto.physics import (
    MEC2,
    RANGE_PROCESS,
    KN_differential_cross,
    lin_att_abs,
    lin_att_compt,
)


def format_aggregation(g: np.ndarray, dim=2) -> np.ndarray:
    """
    Take a 3-dimensional tensor of quality features (4th dimension is features)
    and reorganize it so that it can be used for training [((i,j) by k): features]

    Takes i by j by k and transitions it to either k by i*j, i*k by j, or j*k by
    i
    """
    if len(g.shape) == 3:
        return np.moveaxis(g, dim, -2).reshape((g.shape[0] ** 2, g.shape[0]))
    return np.moveaxis(g, dim, -2).reshape((g.shape[0] ** 2, g.shape[0], g.shape[-1]))


def format_three_aggregations(g: np.ndarray) -> np.ndarray:
    """
    Format all three possible aggregations and combine them into a single form
    """
    out = np.stack(
        (
            format_aggregation(g, dim=0),
            format_aggregation(g, dim=1),
            format_aggregation(g, dim=2),
        ),
        axis=-2,
    )
    return out


def zero_connections(mat: np.ndarray, val: int = 0) -> np.ndarray:
    """
    Zero out connections for a connectivity matrix.

    No connections from a node to itself and no connections back to the origin
    """
    out = np.copy(mat)
    out[np.diag_indices(mat.shape[0])] = val
    out[:, 0] = val
    return out


def labels_to_dict(cluster_indices: Iterable) -> Dict[int, List[int]]:
    """
    Convert a vector of cluster labels to a dict of those clusters.

    Example:
    >>>labels_to_dict([1,2,2,2,8,8,8,8,9,9])
    {1: [1], 2: [2, 3, 4], 3: [5, 6, 7, 8], 4: [9, 10]}
    """
    index_map = {}
    cluster_count = 1
    cluster_dict = {}
    for i, index in enumerate(cluster_indices):
        if index in index_map:
            cluster_dict[index_map[index]].append(i + 1)
        else:
            index_map[index] = cluster_count
            cluster_count += 1
            cluster_dict[index_map[index]] = [i + 1]
    return cluster_dict


# def expand_p_to_g(p : np.ndarray, method='mult', q : np.ndarray = None) -> np.ndarray:
#     """
#     Expand the degree two tensor to the degree three tensor
#     """
#     N = p.shape[0]
#     g = np.zeros((N,N,N))
#     if q is None:
#         q = p
#     if method == 'mult':
#         for (i,j,k) in product(range(N), range(N), range(N)):
#             g[i,j,k] = p[i,j] * q[j,k]
#     elif method == 'sum':
#         for (i,j,k) in product(range(N), range(N), range(N)):
#             g[i,j,k] = p[i,j] + q[j,k]
#     return g


def expand_p_to_g(p: np.ndarray, method="mult", q: np.ndarray = None) -> np.ndarray:
    """
    Expand the degree two tensor to the degree three tensor
    """
    if q is None:
        q = p

    if method == "mult":
        g = (p[:, :, np.newaxis] * q[np.newaxis, :, :]).astype(p.dtype)
    elif method == "sum":
        g = p[:, :, np.newaxis] + q[np.newaxis, :, :]
    else:
        raise ValueError("Invalid method. Use 'mult' or 'sum'.")

    return g


def get_grade_features(
    event: Event, subset: List = None, start_point: int = 0
) -> np.ndarray:
    """
    Use built in libraries to compute the grade tensor quickly.

    ## Parameters
    - `event:gamma_ray.Event` : The gamma ray event containing interactions
    - `subset:List` : The indices of interactions that should be considered
    - `inner_radius:float` : The inner radius of the detector
    ## Outputs
    Outputs many values that could be used for evaluating transitions:
    0. P_en: violation of conservation of energy
    1. P_en_continue: energy available to continue
    2. log_P_ij: -log Probability of penetration from `i` to `j` and Compton scattering there
    3. log_P_jk: -log Probability of penetration from `j` to `k` and Compton or absorption there
    4. P_angle: -log Probability of scattering angle `ijk`
    5. E_ijk: Energy `ijk`
    6. E_ijk_mj: Energy `ijk - e_j`
    7. E_ijk_mj_mk: Energy `ijk - e_j - ek`
    8. D_ij: Distance `ij`
    9. D_jk: Distance `jk`
    10. one_minus_cosine_ijk: `1 - cos theta_ijk`
    11. two_int_one_minus_cos_theo: `1 - cos theta_theo_ij` theoretical scattering angle
        assuming only two interactions
    12. square_cos_diff: difference between actual and theoretical cosines
    13. `e_i`: energy at i
    14. `e_j`: energy at j
    15. `e_k`: energy at k
    """
    if subset is None:
        subset = np.arange(len(event.points))
    else:
        if start_point not in subset:
            subset = [0] + list(subset)
        subset = np.array(subset)
    N = len(subset)
    # points = event.point_matrix[subset]
    energies = np.reshape(event.energy_matrix[subset], (N,))
    d_ij = event.ge_distance[subset[:, np.newaxis], subset]

    E_ijk = event.tango_estimates[
        subset[:, np.newaxis, np.newaxis],
        subset[np.newaxis, :, np.newaxis],
        subset[np.newaxis, np.newaxis, :],
    ]
    E_ijk_mj = E_ijk - energies[np.newaxis, :, np.newaxis]
    E_ijk_mj_mk = E_ijk_mj - energies[np.newaxis, np.newaxis, :]

    one_minus_cosine_ijk = (
        1
        - event.cos_act[
            subset[:, np.newaxis, np.newaxis],
            subset[np.newaxis, :, np.newaxis],
            subset[np.newaxis, np.newaxis, :],
        ]
    )

    mu_compt_ijk = lin_att_compt(E_ijk)
    log_P_ij = mu_compt_ijk * d_ij[:, :, np.newaxis]

    mu_abs_ijk_at_k = lin_att_abs(E_ijk_mj)
    mu_compt_ijk_at_k = lin_att_compt(E_ijk_mj)
    log_P_jk = (mu_abs_ijk_at_k + mu_compt_ijk_at_k) * d_ij[np.newaxis, :, :]

    P_angle = np.zeros(E_ijk.shape)
    ind = log_P_ij > 0
    P_angle[ind] = -np.log(
        KN_differential_cross(
            E_ijk[ind],
            one_minus_cosine_ijk[ind],
            E_ijk_mj[ind],
            sigma_compt=mu_compt_ijk[ind] / RANGE_PROCESS,
            integrate=True,
        )
    )

    P_en = np.maximum(-E_ijk_mj_mk, 0) / MEC2  # Violation of energy conservation
    P_en_continue = np.maximum(E_ijk_mj_mk, 0) / MEC2  # Energy left to continue on

    D_ij = np.tile(np.expand_dims(d_ij, axis=2), (1, 1, N))
    D_jk = np.tile(np.expand_dims(d_ij, axis=0), (N, 1, 1))
    e_i = np.tile(np.expand_dims(energies, axis=(1, 2)), (1, N, N))
    e_j = np.tile(np.expand_dims(energies, axis=(0, 2)), (N, 1, N))
    e_k = np.tile(np.expand_dims(energies, axis=(0, 1)), (N, N, 1))
    two_int_one_minus_cos_theo = MEC2 / e_k - MEC2 / (e_k + e_j)
    two_int_one_minus_cos_theo[e_k == 0] = 0

    square_cosine_diff = np.square(one_minus_cosine_ijk - two_int_one_minus_cos_theo)

    for P in (
        P_en,
        P_en_continue,
        log_P_ij,
        log_P_jk,
        P_angle,
        E_ijk,
        E_ijk_mj,
        E_ijk_mj_mk,
        D_ij,
        D_jk,
        one_minus_cosine_ijk,
        two_int_one_minus_cos_theo,
        square_cosine_diff,
        e_i,
        e_j,
        e_k,
    ):
        for j in range(N):  # Cannot visit the same point twice
            P[j, j, :] = 0
            P[j, :, j] = 0
            P[:, j, j] = 0
        P[:, :, 0] = 0  # Cannot return to origin
        P[:, 0, :] = 0  # Cannot return to origin

    return np.stack(
        (
            P_en,
            P_en_continue,
            log_P_ij,
            log_P_jk,
            P_angle,
            E_ijk,
            E_ijk_mj,
            E_ijk_mj_mk,
            D_ij,
            D_jk,
            one_minus_cosine_ijk,
            two_int_one_minus_cos_theo,
            square_cosine_diff,
            e_i,
            e_j,
            e_k,
        ),
        axis=-1,
    )


# We create this data with the following ideas in mind: - We can train a machine
# learning model to distinguish between absorption interactions and scattering
# interactions:
#     - We do this by selecting data from true absorptions and true scatters and
#       training a classifier on it. It is challenging to select additional data
#       outside of true absorptions and true scatters for this case because we
#       cannot easily evaluate what some false series of interactions should be
#       classified as.
# - We can train a machine learning model to distinguish between true scattering
#   events and false scattering events.
#     - We have to decide whether false scattering events within a cluster should
#       be included (for the sake of clustering, we do not really care; for the
#       sake of ordering, we care).
#     - This model should output a degree three transition tensor indicating if
#       i->j->k is a proper scatter.
# - We can get, from the degree three tensor, a degree two transition matrix
#     - This matrix could possibly be used for clustering
#     - This matrix can be easily used for a new FOM


# def get_2d_transition_grade_estimate(event: Event,
#                                      subset: List = None,
#                                      start_point:int=0) -> np.ndarray:
#     """
#     Get an estimate for the transition grade from the full matrix
#     """
#     features = get_grade_features(event, subset, start_point=start_point)
