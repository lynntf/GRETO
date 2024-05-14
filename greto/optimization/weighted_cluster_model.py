"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Weighted clustering problem

Written by Dominic Yang
"""
from itertools import combinations
from typing import Dict

import pyomo.environ as pyenv

from greto.event_class import Event


def create_weighted_cluster_problem(
    event: Event, cluster_FOM_mapping: Dict, singleton_weight: float = 0.5
) -> pyenv.ConcreteModel:
    """
    Create a problem to determine the clustering which minimizes the sum
    FOMs of clusters while also maximizing the coverage of the clusterings.
    The cluster mapping needs to include all singleton clusters.
    """
    model = pyenv.ConcreteModel()
    model.cluster_idxs = pyenv.RangeSet(len(cluster_FOM_mapping))
    model.clusters = list(cluster_FOM_mapping)
    model.FOMs = list(cluster_FOM_mapping.values())

    # Assign a weight to penalize picking singletons
    for i in model.cluster_idxs:  # pylint: disable=E1133
        if len(model.clusters[i - 1]) == 1:
            model.FOMs[i - 1] = singleton_weight

    model.u = pyenv.Var(model.cluster_idxs, within=pyenv.Binary)
    model.intersection_constraints = pyenv.ConstraintList()
    for i, j in combinations(model.cluster_idxs, 2):
        if len(set(model.clusters[i - 1]) & set(model.clusters[j - 1])) > 0:
            model.intersection_constraints.add(model.u[i] + model.u[j] <= 1)

    model.full_coverage_constraint = pyenv.Constraint(
        expr=sum(len(model.clusters[i - 1]) * model.u[i] for i in model.cluster_idxs)
        == len(event.hit_points)
    )  # pylint: disable=E1133

    expr = sum(
        model.FOMs[i - 1] * model.u[i] for i in model.cluster_idxs
    )  # pylint: disable=E1133
    model.obj = pyenv.Objective(expr=expr, sense=pyenv.minimize)

    return model
