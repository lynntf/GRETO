"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Set up a hierarchical optimization problem for solution using MIP solver with Pyomo.

Written by Dominic Yang
"""
from itertools import combinations
from typing import List

import numpy as np
import pyomo.environ as pyenv

from greto.cluster_utils import get_all_clusters, in_peak
from greto.event_class import Event


def create_hierarchical_cluster_problem(
    event: Event, observed_energies: List[float], epsilon: float = 0.005
):
    """
    Construct and return the hierarchical cluster problem for this event
    with the given observed_energies.

    Args:
        event (Event): A gamma ray event consisting of an id and interactions
        observed_energies (list[float]): The energies of known peaks
        epsilon (float): The tolerance for a energy in a peak
    """
    linkage = event.linkage_array()
    all_clusters = get_all_clusters(linkage, len(event.hit_points))
    e_sums = event.energy_sums(all_clusters)
    peak_clusters = [
        cluster
        for (i, cluster) in all_clusters.items()
        if in_peak(e_sums[i], np.array(observed_energies), epsilon)
    ]

    # If no clusters are at peak energies, then no model can be created
    if len(peak_clusters) == 0:
        return None
    model = pyenv.ConcreteModel()
    model.peak_clusters = peak_clusters
    model.cluster_idxs = pyenv.RangeSet(len(peak_clusters))
    model.u = pyenv.Var(model.cluster_idxs, within=pyenv.Binary)
    model.intersection_constraints = pyenv.ConstraintList()

    for i, j in combinations(model.cluster_idxs, 2):
        if len(set(peak_clusters[i - 1]) & set(peak_clusters[j - 1])) > 0:
            model.intersection_constraints.add(model.u[i] + model.u[j] <= 1)

    eps = 1 / sum(len(c) for c in peak_clusters)

    expr = sum(
        (1 + eps * len(peak_clusters[i - 1])) * model.u[i] for i in model.cluster_idxs
    )  # pylint: disable=E1133
    model.obj = pyenv.Objective(expr=expr, sense=pyenv.maximize)

    return model


def hierarchical_cluster(
    event, S, observed_energies, alpha=0.4, epsilon=0.005, solver="cplex"
):
    """
    Construct the clustering given by solving the hierarchical cluster
    problem and then performing a standard clustering on the remaining
    points.

    Arguments:
        event Event: gamma ray event
        S int: number of clusters
        observed_energies list: energies that we would like the data to reproduce
        alpha float: angular distance to join clusters
        epsilon float: width of V shape objective around energies (in MeV)
        solver string: MIP solver name
    """
    # If just 1 point, put it in a cluster by itself
    if len(event.hit_points) == 1:
        return event.cluster(1)
    model = create_hierarchical_cluster_problem(event, observed_energies, epsilon)

    if model is not None:
        solver = pyenv.SolverFactory(solver)
        solver.options["threads"] = 1
        solver.solve(model, tee=True)
        clusters = sorted(
            [
                sorted(model.peak_clusters[i - 1])
                for i in model.u
                if pyenv.value(model.u[i]) > 0.5
            ]
        )
    else:
        # If it is None, then, the clustering found no peak clusters
        clusters = []

    remaining_points = [
        i
        for i in range(1, len(event.points))
        if all(i not in cluster for cluster in clusters)
    ]
    remainder_event = event.subset(remaining_points)
    remainder_clusters = remainder_event.cluster_linkage(alpha, S - len(clusters))

    # Now we reindex the points back into original coordinates
    all_clusters = dict(enumerate(clusters))
    for _, cluster in remainder_clusters.items():
        reindexed_cluster = [remaining_points[k - 1] for k in cluster]
        all_clusters[max(all_clusters, default=0) + 1] = reindexed_cluster
    return all_clusters
