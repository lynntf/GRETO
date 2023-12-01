"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Create an abstract model for the piecewise tsp problem

Written by Dominic Yang
"""
import pyomo.environ as pyenv

model = pyenv.AbstractModel()

model.N = pyenv.Param()

model.I = pyenv.RangeSet(model.N)
model.I0 = pyenv.RangeSet(0, model.N)

def pairs_rule(model):
    return set((i, j) for i in model.I0 for j in model.I0 if i != j)
model.pairs = pyenv.Set(dimen=2, initialize=pairs_rule)

def angles_rule(model):
    return set([(i, j, k) for i in model.I0 for j in model.I0 for k in model.I0 if i != j and j != k])
model.angles = pyenv.Set(dimen=3, initialize=angles_rule)
model.I_minus = pyenv.RangeSet(model.N-1)
model.dims = pyenv.RangeSet(3)
model.x = pyenv.Param(model.I0, model.dims)
model.e = pyenv.Param(model.I)

def y_rule(model, i, j, dim):
    return model.x[j,dim] - model.x[i,dim]
model.y = pyenv.Param(model.pairs, model.dims, initialize=y_rule)

def c_rule(model, i, j, k):
    return (sum(model.y[i,j,dim]*model.y[j,k,dim] for dim in model.dims)
            / (pyenv.sqrt(sum(model.y[i,j,dim]**2 for dim in model.dims))
               * pyenv.sqrt(sum(model.y[j,k,dim]**2 for dim in model.dims))))

model.c = pyenv.Param(model.angles, initialize=c_rule)

model.m0c2 = pyenv.Param(initialize=0.511)
def e0_rule(model):
    return sum(model.e[i] for i in model.I)
model.E0 = pyenv.Param(initialize=e0_rule)

model.c_theo = pyenv.Var(model.I_minus, within=pyenv.Reals, bounds=(-1, 1))
model.c_act = pyenv.Var(model.I_minus, within=pyenv.Reals, bounds=(-1, 1))
model.z = pyenv.Var(model.I0, model.I0, within=pyenv.Binary)

model.w_index = model.angles * model.I_minus
model.w = pyenv.Var(model.w_index, bounds=(0, 1))
model.E = pyenv.Var(model.I0, within=pyenv.NonNegativeReals, bounds=(0, 1.4))
model.E_inv = pyenv.Var(model.I0, within=pyenv.NonNegativeReals)

def each_point_in_path_rule(model, i):
    return sum(model.z[i,j] for j in model.I0) == 1
model.each_point_in_path = pyenv.Constraint(model.I0, rule=each_point_in_path_rule)

def one_j_in_path_rule(model, j):
    return sum(model.z[i,j] for i in model.I0) == 1
model.one_j_in_path = pyenv.Constraint(model.I0, rule=one_j_in_path_rule)

def full_energy_start_rule(model):
    return model.E[0] == model.E0
model.full_energy_start = pyenv.Constraint(rule=full_energy_start_rule)

def energy_drop_rule(model, j):
    return model.E[j] == model.E[j-1] - sum(model.e[i] * model.z[i,j] for i in model.I)
model.energy_drop = pyenv.Constraint(model.I, rule=energy_drop_rule)

def theo_angle_rule(model, i):
    return model.c_theo[i] == 1 - model.m0c2* (model.E_inv[i] - model.E_inv[i-1])
model.theo_angle = pyenv.Constraint(model.I_minus, rule=theo_angle_rule)

def origin_start_rule(model):
    return model.z[0,0] == 1
model.origin_start = pyenv.Constraint(rule=origin_start_rule)

def w_rule_1(model, i, j, k, m):
    return model.w[i,j,k,m] <= model.z[i,m-1]
model.w_1 = pyenv.Constraint(model.w_index, rule=w_rule_1)

def w_rule_2(model, i, j, k, m):
    return model.w[i,j,k,m] <= model.z[j,m]
model.w_2 = pyenv.Constraint(model.w_index, rule=w_rule_2)

def w_rule_3(model, i, j, k, m):
    return model.w[i,j,k,m] <= model.z[k,m+1]
model.w_3 = pyenv.Constraint(model.w_index, rule=w_rule_3)

def w_rule_4(model, i, j, k, m):
    return model.w[i,j,k,m] >= model.z[i,m-1] + model.z[j,m] + model.z[k,m+1] - 2
model.w_4 = pyenv.Constraint(model.w_index, rule=w_rule_4)

def act_angle_rule(model, l):
    return model.c_act[l] == sum(model.c[i,j,k] * model.w[i,j,k,l] for (i, j, k) in model.angles)
model.act_angle = pyenv.Constraint(model.I_minus, rule=act_angle_rule)

def fom_rule(model):
    return (1/(model.N-1))* sum((model.c_theo[i] - model.c_act[i])**2 for i in model.I_minus)
model.FOM = pyenv.Objective(rule=fom_rule, sense=pyenv.minimize)

def augmented_FOM_rule(model):
    return (1/(model.N-1))* sum((model.c_theo[i] - model.c_act[i])**2 for i in model.I_minus)

def warmstart(instance, permutation):
    # Initialze z
    # The first point in the path is the origin
    instance.z[0,0] = 1
    for i in instance.I:
        instance.z[i,0] = 0
        instance.z[0,i] = 0

    for i, p_idx in enumerate(permutation):
        # it is p_idx+1 because the zeroth index is implicit in permutation and not included
        instance.z[p_idx+1,i+1] = 1
        for j in instance.I:
            if j != i+1:
                instance.z[p_idx+1,j] = 0

    # Initialize E
    instance.E[0] = instance.E0.value
    for i, p_idx in enumerate(permutation):
        instance.E[i+1] = round(instance.E[i].value - instance.e[p_idx+1], 3)

    # Initialize w
    for (i, j, k) in instance.angles:
        for m in instance.I_minus:
            if instance.z[i,m-1].value > 1-1e6 and instance.z[j,m].value > 1e-6 and instance.z[k,m+1].value > 1e-6:
                instance.w[i,j,k,m] = 1
            else:
                instance.w[i,j,k,m] = 0

    # Initialize c_act, c_theo
    for i in instance.I_minus:
        m = i
        instance.c_act[m] = (sum(instance.c[i,j,k]*instance.w[i,j,k,m].value for (i,j,k) in instance.angles))
        instance.c_theo[i] = 1 - instance.m0c2.value * (1/instance.E[i].value - 1/instance.E[i-1].value)


def add_piecewise_rule(instance, xs, pw_repn='SOS2'):
    ys = [1/x for x in xs]
    for i in range(instance.N.value):
        attr_name = f'p{i}'
        setattr(instance, attr_name, pyenv.Piecewise(instance.E_inv[i], instance.E[i], pw_pts=xs,
                                                    pw_constr_type='EQ', f_rule=ys, pw_repn=pw_repn))
