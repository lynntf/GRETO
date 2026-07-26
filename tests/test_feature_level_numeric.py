from __future__ import annotations

import numpy as np
from greto.fast_features.feature_level import build_feature_specs


class DummyPerm:
    def __init__(self):
        self.res_sum_geo = np.array([1.0, 2.0, 3.0])
        self.res_cos = np.array([0.1, 0.2, 0.3])
        self.res_cos_v = np.array([0.1, 0.2, 0.3])
        self.res_cos_sigma = np.array([1.0, 1.0, 1.0])
        self.linear_attenuation_abs = np.array([0.2, 0.3, 0.5])
        self.lin_mu_total = np.array([1.0, 1.0, 1.0])
        self.linear_attenuation_compt = np.array([0.05, 0.1, 0.2])
        self.ge_distance_perm = np.array([1.0, 2.0, 3.0])
        self.distance_perm = np.array([0.5, 1.0, 1.5])
        self.energies_perm = np.array([5.0, 2.0, 1.0])
        self.energy_sum = 8.0
        self.tango_estimates_perm = np.array([0.1, 0.2, 0.3])
        self.tango_estimates_sigma_perm = np.array([1.0, 1.0, 1.0])
        self.escape_probability = 0.25


def find_spec(specs, name):
    for s in specs:
        if s.name == name:
            return s
    raise KeyError(name)


def run():
    pc = DummyPerm()
    specs = build_feature_specs(perm_calc=pc, Nmi=3)

    # rsg_sum_1 == sum of res_sum_geo
    s = find_spec(specs, "rsg_sum_1")
    assert abs(s.compute_fn() - np.sum(pc.res_sum_geo)) < 1e-12

    # rc_sum_1 == sum of res_cos
    s = find_spec(specs, "rc_sum_1")
    assert abs(s.compute_fn() - np.sum(pc.res_cos)) < 1e-12

    # cross_abs_final == last of linear_attenuation_abs
    s = find_spec(specs, "cross_abs_final")
    assert abs(s.compute_fn() - pc.linear_attenuation_abs[-1]) < 1e-12

    # p_abs_final == linear_attenuation_abs[-1] / lin_mu_total[-1]
    s = find_spec(specs, "p_abs_final")
    assert abs(s.compute_fn() - (pc.linear_attenuation_abs[-1] / pc.lin_mu_total[-1])) < 1e-12

    # first_is_not_largest: energies_perm -> no later energy is larger than the first
    s = find_spec(specs, "first_is_not_largest")
    assert s.compute_fn() is False

    # -log_escape_probability
    s = find_spec(specs, "-log_escape_probability")
    assert abs(s.compute_fn() - (-np.log(pc.escape_probability + 1e-16))) < 1e-12

    print("numeric tests: OK")


if __name__ == "__main__":
    run()
