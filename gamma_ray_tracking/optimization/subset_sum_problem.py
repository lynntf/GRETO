"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Subset sum problem for clustering using energy.

Model written by Dominic Yang
Model updated for multiple single solves by Thomas Lynn
TODO - multiple single solve version should be a single model with a subproblem?
"""
from itertools import combinations
from typing import Iterable

import numpy as np
import pyomo.environ as pyenv

from ..event_class import Event


def create_subset_sum_problem(event:Event, energies:Iterable[float],
                              point_subset:Iterable[int]=None,
                              S:int=2, epsilon:float=0.01, M:int=8):
    """
    Create the subset sum problem model to be passed to the MIP solver
    """
    model = pyenv.ConcreteModel()

    model.event = event

    # Initialize sets
    if point_subset is None:
        model.points = pyenv.RangeSet(len(event.hit_points))
    else:
        model.points = pyenv.Set(initialize=point_subset)
    model.clusters = pyenv.RangeSet(S)

    # Initialize variables
    # UB = 1e5
    # UB = 100
    UB = 2*max(sum(event.points[i].e for i in model.points), 2)
    # UB = max(max(event.points[i].e for i in model.points), max(energies) + epsilon) + epsilon
    model.u = pyenv.Var(model.points, model.clusters, within=pyenv.Binary)
    model.E = pyenv.Var(model.clusters, within=pyenv.NonNegativeReals, bounds=(0, UB))
    model.cost = pyenv.Var(model.clusters, within=pyenv.NonNegativeReals)

    # clustering rules: each point should be in a cluster
    model.each_point_in_cluster = pyenv.ConstraintList()
    for i in model.points:
        model.each_point_in_cluster.add(sum(model.u[i,s] for s in model.clusters) == 1)

    # Max cluster size <= 8: cannot have clusters with size larger than 8 interactions
    model.max_cluster_size = pyenv.ConstraintList()
    for s in model.clusters: # pylint: disable=E1133
        model.max_cluster_size.add(sum(model.u[i,s] for i in model.points) <= M)

    # Energy computation
    model.energy_sum = pyenv.ConstraintList()
    for s in model.clusters: # pylint: disable=E1133
        model.energy_sum.add(sum(model.u[i,s] * event.points[i].e
                                for i in model.points) == model.E[s])

    # Cost computation
    domain_pts = [0]
    range_pts  = [1]
    for energy in energies:
        # We add a little v around each energy level
        domain_pts.extend([energy-epsilon, energy, energy+epsilon])
        # range_pts.extend([1, 0, 1])
    domain_pts = sorted(domain_pts)
    for domain_pt in domain_pts[1:]:
        min_dist = UB
        for energy in energies:
            min_dist = min(min_dist, *[abs(domain_pt - energy)])
        range_val = min(min_dist/epsilon, 1)
        range_pts.append(range_val)
    # Upper bound on domain
    domain_pts.append(UB)
    range_pts.append(1)
    # domain_pts = [0, 1, UB]
    # range_pts = [1, 0, UB]
    # print(domain_pts)
    # print(range_pts)

    model.cost_computation = pyenv.Piecewise(
          model.clusters,
          model.cost, model.E,
          pw_pts=domain_pts,
          pw_repn='SOS2',
          pw_constr_type = 'EQ',
          f_rule = range_pts,
          force_pw=True)

    model.obj = pyenv.Objective(expr=sum(model.cost[s] for s in model.clusters),
                                sense=pyenv.minimize)

    return model

def create_subset_sum_single_problem(event:Event, energy:float,
                                     point_subset:Iterable[int]=None,
                                     S:int=2, M:int=8):
    """
    Create the subset sum problem model to be passed to the MIP solver for a single energy
    """
    model = pyenv.ConcreteModel()

    model.event = event

    # Initialize sets
    if point_subset is None:
        model.points = pyenv.RangeSet(len(event.hit_points))
    else:
        model.points = pyenv.Set(initialize=point_subset)
    model.clusters = pyenv.RangeSet(S)

    # Initialize variables
    UB = 2*max(sum(event.points[i].e for i in model.points), 2)
    model.u = pyenv.Var(model.points, model.clusters, within=pyenv.Binary)
    model.E = pyenv.Var(model.clusters, within=pyenv.NonNegativeReals, bounds=(0, UB))
    model.cost = pyenv.Var(model.clusters, within=pyenv.NonNegativeReals)

    # clustering rules: each point should be in a cluster
    model.each_point_in_cluster = pyenv.ConstraintList()
    for i in model.points:
        model.each_point_in_cluster.add(sum(model.u[i,s] for s in model.clusters) == 1)

    # Max cluster size <= 8: cannot have clusters with size larger than 8 interactions
    model.max_cluster_size = pyenv.ConstraintList()
    for s in model.clusters: # pylint: disable=E1133
        model.max_cluster_size.add(sum(model.u[i,s] for i in model.points) <= M)

    # Energy computation
    model.energy_sum = pyenv.ConstraintList()
    for s in model.clusters: # pylint: disable=E1133
        model.energy_sum.add(sum(model.u[i,s] * event.points[i].e
                                for i in model.points) == model.E[s])

    model.obj = pyenv.Objective(expr=abs(model.E[1] - energy),
                                sense=pyenv.minimize)

    return model

def subset_sum_stack(event:Event, observed_energies:list,
                     point_subset:list = None, S:int=2,
                     epsilon:float=0.05, M:int=8,
                     solver_name:str='scip', debug:bool=False) -> dict:
    """
    Do multiple subset sum solves, one for each potential energy, with two
    clusters, one true cluster and one junk cluster. After constructing all
    candidate clusters, select the one with the best FOM. Uses AGATA FOM for
    ordering and GRETINA FOM for validation.
    """
    solver = pyenv.SolverFactory(solver_name)
    if point_subset is None:
        point_subset = list(range(1, len(event.points)))
    tracks = {}
    tracks_id = 1
    continue_flag = True
    while continue_flag and len(point_subset) > 0:
        continue_flag = False
        potential_tracks = {}
        for energy in observed_energies:
            model = create_subset_sum_single_problem(event, energy,
                                                     point_subset=point_subset,
                                                     S=S, M=M)
            solver.solve(model, tee=False)
            track = get_clusters(model)
            if debug:
                print(f'energy {energy} track {track}')
                print(f'          energies {event.energy_sums(track)}')
            energies = event.energy_sums(track)
            for track_id, track_indices in track.items():
                if abs(energies[track_id] - energy) < epsilon:
                    # cluster = event.semi_greedy(track_indices, fom_method='agata')
                    # potential_tracks[tuple(cluster)] = event.FOM(cluster, fom_method='angle')
                    potential_tracks[tuple(track_indices)] = abs(energies[track_id] - energy)
        best_fom = np.inf
        for track, fom in potential_tracks.items():
            if fom < best_fom:
                best_fom = fom
                best_track = track
                continue_flag = True
        if continue_flag and len(potential_tracks) > 0:
            tracks[tracks_id] = best_track
            tracks_id += 1
            for index in best_track:
                point_subset.remove(index)
    if len(point_subset) > 0:
        tracks[tracks_id] = point_subset
    return tracks


def add_local_distance_constraints(instance, clusters, du):
    """
    Add constraints on distances
    """
    u_hat = {}
    for i in instance.points:
        for s in instance.clusters:
            if i in clusters[s]:
                u_hat[i,s] = 1
            else:
                u_hat[i,s] = 0

    instance.local_distance = pyenv.Constraint(rule=
            sum(1 - instance.u[i,s] for i in instance.points
                for s in instance.clusters if u_hat[i,s] == 1)
            + sum(instance.u[i,s] for i in instance.points
                  for s in instance.clusters if u_hat[i,s] == 0) <= du)


def add_cluster_max_distance_constraints(instance, distance):
    """
    Adds constraints which prohibit putting two points in the same cluster
    which have an angular distance greater than distance.
    """
    instance.distance_constraints = pyenv.ConstraintList()

    for (i, j) in combinations(instance.points, 2):
        pi, pj = instance.event.points[i], instance.event.points[j]
        angle = np.arccos(pi.x @ pj.x / (np.linalg.norm(pi.x) * np.linalg.norm(pj.x)))
        if angle > distance:
            for s in instance.clusters:
                instance.distance_constraints.add(instance.u[i,s] + instance.u[j,s] <= 1)


def get_clusters(model):
    """
    Get the clusters from the model
    """
    clusters = {}
    for s in model.clusters:
        cluster = []
        for i in model.points:
            try:
                if pyenv.value(model.u[i,s]) > 0.5:
                    cluster.append(i)
            except Exception as e:
                print('****************')
                print(model)
                raise e
        if len(cluster) > 0:
            clusters[s] = cluster
    return clusters
