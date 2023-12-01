"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Original clustering model using the cosine figure of merit.

Written by Dominic Yang
"""
import math
from itertools import combinations
from typing import Iterable, List, Dict

import numpy as np
import pyomo.environ as pyenv

from ..event_class import Event
from ..interaction_class import Interaction


def get_breaks(x0: float, x_max: float, e: float) -> List[float]:
    """
    Get piecewise breakpoints for the gamma-ray tracking problem
    """
    xs = [x0]
    while xs[-1] < x_max:
        x = xs[-1] + np.sqrt(4*xs[-1]**3 * e)
        xs.append(x)
    xs[-1] = x_max
    return xs

def evaluate_piecewise(x: float, xs: Iterable[float], ys: Iterable[float]) -> float:
    """
    Evaluate a linear piecewise function with points xs and values ys
    """
    if len(xs) != len(ys):
        raise ValueError("xs not the same length as ys")
    if x < xs[0] or x > xs[-1]:
        raise ValueError("x is out of bounds of xs")

    for i, xi in enumerate(xs):
        if x < xi:
            index = i
            break

    y = ys[index-1] + (ys[index]-ys[index-1])/(xs[index]-xs[index-1]) * (x - xs[index-1])
    return y


def build_model(event:Event, FOM_method:str='sum', error:str='l2',
                S:int=1, M:int=8, distance:int=0,
                alpha:float=0.2, symmetry:bool=False,
                piecewise_error:float=0.1, x_0:float=0.01,
                x_max:float=10, singleton_penalty:float=0, energy_big_M:bool=False,
                pw_repn:str='SOS2', add_escape_event:bool=False) -> pyenv.ConcreteModel:
    """
    FOM_method: Set to 'avg' to normalize cluster FOMs by size, set to 'sum'
        to not do this.
    error: The error metric used: Options are 'l1' or 'l2'
    S (int): The number of clusters
    M (int): The maximum number of interactions in a cluster
    distance: An integer specifying the type of distance requirements on clusters
        If set to 0, there are no distance requirements
        If set to 1, each pair of points in a cluster are required to have
            angle less than alpha
        2 is invalid
        If set to 3, each pair of subsequent points in an ordering required
            to have angle less than alpha
    alpha (float): The distance threshold used for distance requirements
    piecewise_error (float): The maximum error allowed on the piecewise approx
        of 1/x
    x_0 (float): The lowest energy allowed
    x_max (float): The highest energy allowed
    singleton_penalty (float): The penalty given to singleton clusters
    energy_big_M (bool): Apply bigM rule to energy drop constraints
        instead of compton scattering formula
    pw_repn (str): The type of piecewise representation used, see pyomo docs
        for options
    add_escape_event (bool): If True, adds slack to total energy of the
        event to allow for escape events.
    """
    model = pyenv.ConcreteModel()

    model.N = len(event.hit_points)
    model.Nm1 = pyenv.Param(initialize=model.N-1)
    model.M = pyenv.Param(initialize=M)

    model.I = pyenv.RangeSet(model.N)
    model.I0 = pyenv.RangeSet(0, model.N)
    model.I_minus = pyenv.RangeSet(model.Nm1)
    model.J = pyenv.RangeSet(model.M)
    model.J0 = pyenv.RangeSet(0, model.M)
    model.J_minus = pyenv.RangeSet(M-1)

    def pairs_rule(model):
        return set([(i, j) for i in model.I0 for j in model.I0 if i != j])
    model.pairs = pyenv.Set(dimen=2, initialize=pairs_rule)

    model.angles = event.valid_angles(alpha)
    model.dims = pyenv.RangeSet(3)
    model.x = {}
    model.e = {}
    for i, p in enumerate(event.points):
        for dim in range(1, 4):
            model.x[i,dim] = p.x[dim-1]
        model.e[i] = p.e

    def theta_rule(model, i):
        r = math.sqrt(model.x[i,1]**2 + model.x[i,2]**2 + model.x[i,3]**2)
        return math.acos(model.x[i,3]/r)
    model.theta = pyenv.Param(model.I, initialize=theta_rule)

    def phi_rule(model, i):
        return math.atan2(model.x[i,2], model.x[i,1])
    model.phi = pyenv.Param(model.I, initialize=phi_rule)

    model.S = S
    model.clusters = pyenv.RangeSet(model.S)

    def dx_rule(model, i, j, dim):
        return model.x[j,dim] - model.x[i,dim]
    model.dx = pyenv.Param(model.pairs, model.dims, initialize=dx_rule)

    def c_rule(model, i, j, k):
        return (sum(model.dx[i,j,dim]*model.dx[j,k,dim] for dim in model.dims)
                / (pyenv.sqrt(sum(model.dx[i,j,dim]**2 for dim in model.dims))
                   * pyenv.sqrt(sum(model.dx[j,k,dim]**2 for dim in model.dims))))

    model.c = pyenv.Param(model.angles, initialize=c_rule)

    model.m0c2 = 0.511
    def e0_rule(model):
        return sum(model.e[i] for i in model.I)
    model.E0 = pyenv.Param(initialize=e0_rule)

    model.c_theo = pyenv.Var(model.J_minus, model.clusters,
                             within=pyenv.Reals, bounds=(-1, 1))
    model.c_act = pyenv.Var(model.J_minus, model.clusters,
                            within=pyenv.Reals, bounds=(-1, 1))

    if FOM_method == 'avg':
        model.r_theo = pyenv.Var(model.J_minus, model.J,
                                model.clusters, bounds=(-1, 1))
        model.r_act = pyenv.Var(model.J_minus, model.J,
                                model.clusters, bounds=(-1, 1))

    # y[j,s] = 1 if cluster s has exactly j elements
    model.y = pyenv.Var(model.J0, model.clusters, bounds=(0,1))

    if error == 'l1':
        if FOM_method == 'avg':
            model.error = pyenv.Var(model.J_minus, model.J, model.clusters)
        elif FOM_method == 'sum':
            model.error = pyenv.Var(model.J_minus, model.clusters)

    model.z = pyenv.Var(model.I0, model.J0,
                        model.clusters, within=pyenv.Binary)

    model.w_index = model.angles * model.J_minus * model.clusters
    model.w = pyenv.Var(model.w_index, bounds=(0, 1))
    model.E = pyenv.Var(model.J0, model.clusters,
                        within=pyenv.NonNegativeReals, bounds=(0, 10))
    model.E_inv = pyenv.Var(model.J0, model.clusters,
                            within=pyenv.NonNegativeReals)

    # variable indicating if point i is in cluster s
    model.u = pyenv.Var(model.I0, model.clusters, within=pyenv.Binary)
    # Variable indicating if cluster s has at least i elements
    model.v = pyenv.Var(model.J, model.clusters, bounds=(0,1))

    def each_point_in_path_rule(model, i, s):
        if i == 0:
            # Enforce that origin is in each path
            return sum(model.z[0,j,s] for j in model.J0) == 1
        else:
            # Enforce that if point i in cluster s,
            # then it occurs at some j in path s
            return sum(model.z[i,j,s] for j in model.J) == model.u[i,s]
    model.each_point_in_path = pyenv.Constraint(model.I0, model.clusters,
                                                rule=each_point_in_path_rule)

    def one_j_in_path_rule(model, j, s):
        if j == 0:
            return sum(model.z[i,j,s] for i in model.I0) == 1
        else:
            # Enforce that if cluster s has j elements,
            # there is some point i which is position j in path s
            return sum(model.z[i,j,s] for i in model.I) == model.v[j,s]
    model.one_j_in_path = pyenv.Constraint(model.J0, model.clusters,
                                           rule=one_j_in_path_rule)

    # Constraints which ensure v_j^s is 1 when
    # there are at least j u_i^s=1 and 0 otherwise
    def v_rule_1(model, j, s):
        return sum(model.u[i,s] for i in model.I) >= j * model.v[j,s]
    model.v_1 = pyenv.Constraint(model.J, model.clusters, rule=v_rule_1)

    def v_rule_2(model, j, s):
        lhs = sum(model.u[i,s] for i in model.I)
        rhs = j - 1 + (model.N-j+1)*model.v[j,s]
        return lhs <= rhs
    model.v_2 = pyenv.Constraint(model.J, model.clusters, rule=v_rule_2)

    if add_escape_event:
        model.E_esc = pyenv.Var(model.clusters, bounds=(0, 1.3))
        model.v_esc = pyenv.Var(model.clusters, within=pyenv.Binary)

        # If have an escape event, not all energy is deposited in the cluster
        def full_energy_start_rule(model, s):
            rhs = sum(model.e[i]*model.u[i,s] for i in model.I) + model.E_esc[s]
            return model.E[0,s] == rhs
        model.full_energy_start = pyenv.Constraint(
                model.clusters, rule=full_energy_start_rule)

        # def escape_event_rule(model, s):
        #     return model.E_esc[s] <= 1.3 * model.v_esc[s]

    else:
        # Fix energy in cluster at start to be some of energy deposits in cluster
        def full_energy_start_rule(model, s):
            return model.E[0,s] == sum(model.e[i]*model.u[i,s] for i in model.I)
        model.full_energy_start = pyenv.Constraint(
                model.clusters, rule=full_energy_start_rule)

    # If True, we apply the big M constraints to deactivate the energy drop rule
    # instead of the compton scattering formula. This is easier to bound
    if energy_big_M:
        # Energy drops at each stop by energy in point j
        # only active if v_j+1^s is 1
        def energy_drop_rule_1(model, j, s):
            rhs = (model.E[j-1,s]
                   - sum(model.e[i] * model.z[i,j,s] for i in model.I)
                   + 5*(1 - model.v[j+1,s]))
            return model.E[j,s] <= rhs
        model.energy_drop_1 = pyenv.Constraint(model.J_minus, model.clusters,
                                               rule=energy_drop_rule_1)

        def energy_drop_rule_2(model, j, s):
            rhs = (model.E[j-1,s]
                   - sum(model.e[i] * model.z[i,j,s] for i in model.I)
                   - 5*(1-model.v[j+1,s]))
            return model.E[j,s] >= rhs
        model.energy_drop_2 = pyenv.Constraint(model.J_minus, model.clusters,
                                               rule=energy_drop_rule_2)

        # Compton scattering rule
        def theo_angle_rule(model, j, s):
            rhs = 1 - model.m0c2* (model.E_inv[j,s] - model.E_inv[j-1,s])
            return model.c_theo[j,s] == rhs
        model.theo_angle = pyenv.Constraint(model.J_minus, model.clusters,
                                            rule=theo_angle_rule)

    else:
        # Energy drops at each stop by energy in point j
        def energy_drop_rule(model, j, s):
            rhs = model.E[j-1,s] - sum(model.e[i] * model.z[i,j,s]
                                       for i in model.I)
            return model.E[j,s] == rhs
        model.energy_drop = pyenv.Constraint(model.J, model.clusters,
                                             rule=energy_drop_rule)

        # Compton scattering rule, only active if v_j+1^s is 1
        def theo_angle_rule_1(model, j, s):
            rhs = (1 - model.m0c2* (model.E_inv[j,s] - model.E_inv[j-1,s])
                   + 100*(1 - model.v[j+1,s]))
            return model.c_theo[j,s] <= rhs
        model.theo_angle_1 = pyenv.Constraint(model.J_minus, model.clusters,
                                              rule=theo_angle_rule_1)

        def theo_angle_rule_2(model, j, s):
            rhs = (1 - model.m0c2* (model.E_inv[j,s] - model.E_inv[j-1,s])
                   - 100*(1 - model.v[j+1,s]))
            return model.c_theo[j,s] >= rhs
        model.theo_angle_2 = pyenv.Constraint(model.J_minus, model.clusters,
                                              rule=theo_angle_rule_2)

    # Each path starts at the origin
    def origin_start_rule(model, s):
        return model.z[0,0,s] == 1
    model.origin_start = pyenv.Constraint(
            model.clusters, rule=origin_start_rule)

    # Each cluster includes the origin
    def origin_start_rule_2(model, s):
        return model.u[0,s] == 1
    model.origin_start_2 = pyenv.Constraint(
            model.clusters, rule=origin_start_rule_2)

    # These four constraints ensure w[i,j,k,l,s] is 1
    # if angle i->j->k is used at step l in cluster s
    def w_rule_1(model, i, j, k, m, s):
        return model.w[i,j,k,m,s] <= model.z[i,m-1,s]
    model.w_1 = pyenv.Constraint(model.w_index, rule=w_rule_1)

    def w_rule_2(model, i, j, k, m, s):
        return model.w[i,j,k,m,s] <= model.z[j,m,s]
    model.w_2 = pyenv.Constraint(model.w_index, rule=w_rule_2)

    def w_rule_3(model, i, j, k, m, s):
        return model.w[i,j,k,m,s] <= model.z[k,m+1,s]
    model.w_3 = pyenv.Constraint(model.w_index, rule=w_rule_3)

    def w_rule_4(model, i, j, k, m, s):
        rhs = model.z[i,m-1,s] + model.z[j,m,s] + model.z[k,m+1,s] - 2
        return model.w[i,j,k,m,s] >= rhs
    model.w_4 = pyenv.Constraint(model.w_index, rule=w_rule_4)

    # cos angle formula, active only if v_j+1^s is 1
    def act_angle_rule_1(model, j, s):
        rhs = (sum(model.c[angle] * model.w[angle,j,s] for angle in model.angles)
               + 2*(1 - model.v[j+1,s]))
        return model.c_act[j,s] <= rhs
    model.act_angle_1 = pyenv.Constraint(model.J_minus, model.clusters,
                                         rule=act_angle_rule_1)

    def act_angle_rule_2(model, j, s):
        rhs = (sum(model.c[angle] * model.w[angle,j,s] for angle in model.angles)
               - 2*(1 - model.v[j+1,s]))
        return model.c_act[j,s] >= rhs
    model.act_angle_2 = pyenv.Constraint(model.J_minus, model.clusters,
                                         rule=act_angle_rule_2)

    # Ensure each point is in exactly 1 cluster
    def point_in_cluster_rule(model, i):
        return sum(model.u[i,s] for s in model.clusters) == 1
    model.point_in_cluster = pyenv.Constraint(model.I, rule=point_in_cluster_rule)

    # Ensures each cluster has at least 1 point.
    #def cluster_has_points_rule(model, s):
    #    return sum(model.u[i,s] for i in model.I) >= 1
    #model.cluster_has_points_rule = pyenv.Constraint(model.clusters,
    #                                       rule=cluster_has_points_rule)

    # Construct constraints for handling average FOM formulation
    if FOM_method == 'avg':
        def mccormick_rule_1_act(model, i, j, s):
            return model.r_act[i,j,s] + model.y[j,s] >= 0
        model.mccormick_1_act = pyenv.Constraint(model.J_minus, model.J,
                model.clusters, rule=mccormick_rule_1_act)
        def mccormick_rule_2_act(model, i, j, s):
            return model.y[j,s] - model.r_act[i,j,s] >= 0
        model.mccormick_2_act = pyenv.Constraint(model.J_minus, model.J,
                model.clusters, rule=mccormick_rule_2_act)
        def mccormick_rule_3_act(model, i, j, s):
            return model.c_act[i,s] - model.r_act[i,j,s] + 1 - model.y[j,s] >= 0
        model.mccormick_3_act = pyenv.Constraint(model.J_minus, model.J,
                model.clusters, rule=mccormick_rule_3_act)
        def mccormick_rule_4_act(model, i, j, s):
            return 1 - model.c_act[i,s] - model.y[j,s] + model.r_act[i,j,s] >= 0
        model.mccormick_4_act = pyenv.Constraint(model.J_minus, model.J,
                model.clusters, rule=mccormick_rule_4_act)
        def mccormick_rule_1_theo(model, i, j, s):
            return model.r_theo[i,j,s] + model.y[j,s] >= 0
        model.mccormick_1_theo = pyenv.Constraint(model.J_minus, model.J,
                model.clusters, rule=mccormick_rule_1_theo)
        def mccormick_rule_2_theo(model, i, j, s):
            return model.y[j,s] - model.r_theo[i,j,s] >= 0
        model.mccormick_2_theo = pyenv.Constraint(model.J_minus, model.J,
                model.clusters, rule=mccormick_rule_2_theo)
        def mccormick_rule_3_theo(model, i, j, s):
            return model.c_theo[i,s] - model.r_theo[i,j,s] + 1 - model.y[j,s] >= 0
        model.mccormick_3_theo = pyenv.Constraint(model.J_minus, model.J,
                model.clusters, rule=mccormick_rule_3_theo)
        def mccormick_rule_4_theo(model, i, j, s):
            return 1 - model.c_theo[i,s] - model.y[j,s] + model.r_theo[i,j,s] >= 0
        model.mccormick_4_theo = pyenv.Constraint(model.J_minus, model.J,
                model.clusters, rule=mccormick_rule_4_theo)

        # Constraint enforcing y[j,s] is 1 if there are j active u[i,s]
        def cluster_count_rule(model, s):
            lhs = sum(model.u[i,s] for i in model.I)
            rhs = sum(j*model.y[j,s] for j in model.J0)
            return lhs == rhs
        model.cluster_count = pyenv.Constraint(model.clusters,
                rule=cluster_count_rule)

    # Enforce the relation between v and y
    def v_y_rule(model, j, s):
        if j == 0:
            return 1 - model.v[j+1,s] == model.y[j,s]
        if j < model.M:
            return model.v[j,s] - model.v[j+1,s] == model.y[j,s]
        elif j == model.M:
            return model.v[j,s] == model.y[j,s]
    model.v_y = pyenv.Constraint(model.J0, model.clusters, rule=v_y_rule)

    # Construct the objective function, fom_rule
    if error == 'l1':
        if FOM_method == 'avg':
            def l1_FOM_upper_rule(model, i, j, s):
                lhs = (model.r_theo[i,j,s] - model.r_act[i,j,s])
                return lhs <= model.error[i,j,s]

            def l1_FOM_lower_rule(model, i, j, s):
                lhs = (model.r_theo[i,j,s] - model.r_act[i,j,s])
                return lhs >= -model.error[i,j,s]

            def fom_rule(model):
                return (sum((sum(1/(j-1) * model.error[i,j,s]
                                for j in pyenv.RangeSet(2, model.M))) # pylint: disable=E1133
                           for i in model.J_minus for s in model.clusters)
                    + singleton_penalty*sum(model.y[1,s] for s in model.clusters))
            model.l1_FOM_upper = pyenv.Constraint(model.J_minus, model.J,
                    model.clusters, rule=l1_FOM_upper_rule)
            model.l1_FOM_lower = pyenv.Constraint(model.J_minus, model.J,
                    model.clusters, rule=l1_FOM_lower_rule)
        elif FOM_method == 'sum':
            def l1_FOM_upper_rule(model, j, s):
                return (model.c_theo[j,s] - model.c_act[j,s]) <= model.error[j,s]

            def l1_FOM_lower_rule(model, j, s):
                return (model.c_theo[j,s] - model.c_act[j,s]) >= -model.error[j,s]

            def fom_rule(model):
                return (sum(model.error[j,s] for j in model.J_minus for s in model.clusters)
                    + singleton_penalty*sum(model.y[1,s] for s in model.clusters))
            model.l1_FOM_upper = pyenv.Constraint(
                    model.J_minus, model.clusters, rule=l1_FOM_upper_rule)
            model.l1_FOM_lower = pyenv.Constraint(
                    model.J_minus, model.clusters, rule=l1_FOM_lower_rule)
    elif error == 'l2':
        if FOM_method == 'sum':
            def fom_rule(model):
                term1 = sum((model.c_theo[j,s] - model.c_act[j,s])**2
                            for j in model.J_minus for s in model.clusters)
                term2 = singleton_penalty*sum(model.y[1,s]
                            for s in model.clusters)
                return term1 + term2
        elif FOM_method == 'avg':
            def fom_rule(model):
                return (sum((sum(1/pyenv.sqrt(j-1) * (model.r_act[i,j,s] - model.r_theo[i,j,s])
                                for j in pyenv.RangeSet(2, model.M))**2) # pylint: disable=E1133
                           for i in model.J_minus for s in model.clusters)
                        + singleton_penalty*sum(model.y[1,s] for s in model.clusters))

    model.FOM = pyenv.Objective(rule=fom_rule, sense=pyenv.minimize)

    model.alpha = alpha

    xs = get_breaks(x_0, x_max, piecewise_error)
    model.breaks = xs
    add_piecewise_rule(model, xs, pw_repn=pw_repn)

    if FOM_method == 'avg':
        add_sos1_constraints(model)

    if distance == 1:
        add_distance_rule(model)
    elif distance == 2:
        add_distance_rule2(model)
    elif distance == 3:
        add_distance_rule3(model)

    if symmetry:
        add_cluster_symmetry_constraints(model)

    return model


def get_event_from_model(model:pyenv.Model) -> Event:
    """Create the event from the model"""
    ps = []
    for i in model.I:
        x = []
        for dim in model.dims:
            x.append(model.x[i,dim])
        e = model.e[i]
        ps.append(Interaction(x, e))
    return Event(0, ps)


def warm_start(instance:pyenv.Model, permutations:Dict,
              error:str='l2', FOM_method:str='sum'):
    """Give the model some initial conditions"""

    ys = [1/x for x in instance.breaks]
    xs = [0] + instance.breaks
    ys = [ys[0]] + ys

    for s in instance.clusters:
        if s not in permutations:
            permutations[s] = ()

    for s in instance.clusters:

        # Initialize u
        instance.u[0,s] = 1
        for i in instance.I:
            if i in permutations[s]:
                instance.u[i,s] = 1
            else:
                instance.u[i,s] = 0

        # Initialize v
        for j in instance.J:
            if len(permutations[s]) >= j:
                instance.v[j,s] = 1
            else:
                instance.v[j,s] = 0

        # Initialize z
        # The first point in the path is the origin
        instance.z[0,0,s] = 1
        for i in instance.I:
            instance.z[i,0,s] = 0
        for j in instance.J:
            instance.z[0,j,s] = 0

        for j in instance.J:
            if j < len(permutations[s])+1:
                p_idx = permutations[s][j-1]
                instance.z[p_idx,j,s] = 1
                for i in instance.I0:
                    if i != p_idx:
                        instance.z[i,j,s] = 0
            else:
                for i in instance.I0:
                    instance.z[i,j,s] = 0

        # Initialize E
        for j in instance.J0:
            if j == 0:
                instance.E[j,s] = sum(instance.e[p_idx]
                                      for p_idx in permutations[s])
            elif j < len(permutations[s])+1:
                p_idx = permutations[s][j-1]
                instance.E[j,s] = max(instance.E[j-1,s].value
                                      - instance.e[p_idx], 0)
            else:
                instance.E[j,s] = instance.E[j-1,s].value

        # Initialize E_inv
        for j in instance.J0:
            instance.E_inv[j,s] = evaluate_piecewise(instance.E[j,s].value,
                                                    xs, ys)

        # Initialize w
        for (i, j, k) in instance.angles:
            for m in instance.J_minus:
                if (instance.z[i,m-1,s].value > 1-1e-6
                        and instance.z[j,m,s].value > 1-1e-6
                        and instance.z[k,m+1,s].value > 1-1e-6):
                    instance.w[i,j,k,m,s] = 1
                else:
                    instance.w[i,j,k,m,s] = 0

        # Initialize c_act, c_theo
        for j in instance.J_minus:
            if j < len(permutations[s]):
                instance.c_act[j,s] = (sum(instance.c[angle]*instance.w[angle,j,s].value
                                           for angle in instance.angles))
                instance.c_theo[j,s] = (1 - instance.m0c2.value
                        * (instance.E_inv[j,s].value - instance.E_inv[j-1,s].value))
            else:
                # For larger than this value, there is no constraint on c_act, c_theo
                instance.c_act[j,s] = 1
                instance.c_theo[j,s] = 1

        # Initialize y
        for j in instance.J0:
            if len(permutations[s]) == j:
                instance.y[j,s] = 1
            else:
                instance.y[j,s] = 0

        # Initialize r_act and r_theo
        if FOM_method == 'avg':
            for i in instance.J_minus:
                for j in instance.J:
                    instance.r_act[i,j,s] = instance.c_act[i,s].value * instance.y[j,s].value
                    instance.r_theo[i,j,s] = instance.c_theo[i,s].value * instance.y[j,s].value

            # Initialize error
            if error == 'l1':
                for i in instance.J_minus:
                    for j in instance.J:
                        instance.error[i,j,s] = abs(instance.r_theo[i,j,s].value
                                                    - instance.r_act[i,j,s].value)
        elif FOM_method == 'sum':
            # Initialize error
            if error == 'l1':
                for j in instance.J_minus:
                    instance.error[j,s] = abs(instance.c_theo[j,s].value
                                              - instance.c_act[j,s].value)


        # Initialize SOS2 variables
        for j in instance.J0:
            for i, x in enumerate(xs):
                if instance.E[j,s].value < x:
                    break

            piecewise_attr = getattr(instance, f'p{j}_{s}')
            for k in range(len(xs)):
                piecewise_attr.SOS2_y[k] = 0

            lambda1 = (xs[i] - instance.E[j,s].value) / (xs[i] - xs[i-1])
            lambda2 = 1 - lambda1
            piecewise_attr.SOS2_y[i-1] = lambda1
            piecewise_attr.SOS2_y[i] = lambda2

def add_piecewise_rule(instance:pyenv.Model, xs:List[float], pw_repn='SOS2'):
    """Add piecewise inverse function"""
    ys = [1/x for x in xs]
    xs = [0] + xs
    ys = [ys[0]] + ys

    if pw_repn in ['LOG', 'DLOG']:
        log2 = np.log2(len(xs)-1)
        if not np.isclose(log2, int(log2)):
            num_breaks = 2**(int(log2)+1)+1
            print(num_breaks)
            while len(xs) < num_breaks:
                xs.append(xs[-1]+1)
                ys.append(1/xs[-1])
    for j in instance.J0:
        for s in instance.clusters:
            attr_name = f'p{j}_{s}'
            pw_constr = pyenv.Piecewise(
                    instance.E_inv[j,s], instance.E[j,s], pw_pts=xs,
                    pw_constr_type='EQ', f_rule=ys, pw_repn=pw_repn)
            setattr(instance, attr_name, pw_constr)


def angular_distance(instance:pyenv.Model, i1, i2):
    """Angular distance between interactions"""
    dist = abs(math.acos(math.sin(instance.theta[i1])*math.sin(instance.theta[i2])
                         *math.cos(instance.phi[i1] - instance.phi[i2])
                         + math.cos(instance.theta[i1])*math.cos(instance.theta[i2])))
    return dist

def add_distance_rule(instance:pyenv.Model):
    """
    Make it so in each cluster, points are all within a certain distance from
    each other
    """
    instance.distance_constrs = pyenv.ConstraintList()
    for (i1, i2) in combinations(instance.I, 2):
        dist = angular_distance(instance, i1, i2)
        if dist > instance.alpha:
            for s in instance.clusters:
                instance.distance_constrs.add(instance.u[i1,s] + instance.u[i2,s] <= 1)


def add_distance_rule2(instance:pyenv.Model):
    """Make it so for each point, there is at least one other point in the cluster
    within a certain distance"""
    instance.distance_constrs = pyenv.ConstraintList()
    for i1 in instance.I:
        close_points = []
        for i2 in instance.I:
            if i2 == i1:
                continue
            dist = angular_distance(instance, i1, i2)
            if dist <= instance.alpha:
                close_points.append(i2)

        for s in instance.clusters:
            rhs = instance.y[1,s] + sum(instance.u[i2,s] for i2 in close_points)
            instance.distance_constrs.add(instance.u[i1,s] <= rhs)

def add_distance_rule3(instance: pyenv.Model):
    """For each point, ensure the subsequent point is within a certain distance"""
    instance.distance_constrs = pyenv.ConstraintList()
    for (i1, i2) in combinations(instance.I, 2):
        dist = angular_distance(instance, i1, i2)
        if dist > instance.alpha:
            for s in instance.clusters:
                for j in instance.J_minus:
                    lhs = instance.z[i1,j,s] + instance.z[i2,j+1,s]
                    instance.distance_constrs.add(lhs <= 1)
                    lhs = instance.z[i2,j,s] + instance.z[i1,j+1,s]
                    instance.distance_constrs.add(lhs <= 1)


def add_sos1_constraints(instance: pyenv.Model):
    """Add SOS1 constraints to the model"""
    for s in instance.clusters:
        attr_name = f'SOS{s}'
        expr = sum(instance.y[j,s] for j in instance.J0) == 1
        setattr(instance, attr_name, pyenv.Constraint(rule=expr))


def add_local_distance_rule(instance: pyenv.Model, clusters: Dict, max_distance:float):
    """Add a local distance maximum for clustering"""
    u_hat = {}
    for i in instance.I:
        for s in instance.clusters:
            if i in clusters[s]:
                u_hat[i,s] = 1
            else:
                u_hat[i,s] = 0

    term1 = sum(1 - instance.u[i,s]
                for i in instance.I for s in instance.clusters
                if u_hat[i,s] == 1)
    term2 = sum(instance.u[i,s]
                for i in instance.I for s in instance.clusters
                if u_hat[i,s] == 0)
    instance.local_distance = pyenv.Constraint(rule=(term1 + term2 <= max_distance))


def add_cluster_symmetry_constraints(instance:pyenv.Model):
    """
    Constraints added to ensure that permutations of clusters are not also solutions
    """
    instance.symmetry_constrs = pyenv.ConstraintList()
    for i in range(2, instance.N.value+1):
        for j in range(2, instance.S.value+1):
            # Ensures that in cluster j,...,S, can only use point i if
            # a point with a lower index is used cluster j-1
            instance.symmetry_constrs.add(
                    sum(instance.u[i,s] for s in range(j, instance.S.value+1))
                    <= sum(instance.u[p,j-1] for p in range(1, i)))
