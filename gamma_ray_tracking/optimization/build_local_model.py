"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

TODO - write pyomo model
Use a completely local version of the FOM in order to take advantage of
precomputation of all relevant values, making solutions much faster/feasible.
values and solve over a smaller space.

In order to make a mostly (where is really counts) local version of the FOM, we
must use TANGO estimates for incoming energy. These estimates are then compared
against partial energy sums. Estimates can be precomputed using <O(n^3)
time/space (triplets of orders, but they are symmetric, so n*(n-1)*(n-2)/2 or
something similar), potentially saving a lot of computation if the resulting
system can be solved quickly. Quantifying the solution scaling will be important
when comparing this to a semi-greedy formulation of direct enumeration
O(n!/(n-k)!) or direct enumeration in general O(n!).
"""
import pyomo.environ as pyenv

from ..event_class import Event

def build_local_model(event:Event, cluster_indices, error='l2'):
    """Use the N^3 transitions (and TANGO estimates) to do clustering"""
    model = pyenv.ConcreteModel()
    model.N = len(cluster_indices) + 1
    model.Nm1 = pyenv.Param(initialize=model.N - 1)
    model.M = pyenv.Param(initialize=M)

    model.I = pyenv.RangeSet(model.N)
    model.I0 = pyenv.RangeSet(0, model.N)
    model.I_minus = pyenv.RangeSet(model.Nm1)
    model.J = pyenv.RangeSet(model.M)
    model.J0 = pyenv.RangeSet(0, model.M)
    model.J_minus = pyenv.RangeSet(M-1)
    raise NotImplementedError("This model is not complete")