"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

FOM tools
"""
from __future__ import annotations
from copy import deepcopy
from itertools import permutations
from typing import Callable, Dict, Hashable, Iterable, List, Tuple

import numpy as np
from scipy import integrate

from . import default_config
from .detector_config_class import DetectorConfig
from .event_class import Event
from .geometry import cone
from .interaction_class import Interaction
from .physics import MEC2, RANGE_PROCESS, cos_theor, lin_att_total, theta_theor

# %% Clustered FOM
def cluster_FOM(event:Event, clusters:Dict[Hashable, Iterable[int]], **FOM_kwargs):
    """
    Compute the FOM for each of the passed in clusters.

    Args:
        event (Event): A gamma-ray coincidence event
        clusters (Dict): A clustering of the interaction points in
            this event.
        FOM_kwargs (Dict): kwargs for the FOM to be used
    """
    return {s: FOM(event, cluster, **FOM_kwargs)
            for (s, cluster) in clusters.items()}

#%% FOM backbone

def FOM(event: Event, permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        estimate_start_energy: bool = False,
        normalize_start_energy_estimate: bool = False,
        accept_max: bool = True,
        min_excess: float = 0.1,
        max_excess: float = MEC2,
        eres: float = 1e-3,
        Nmi: int = None,
        singles_method: str = 'depth',
        singles_penalty_min: float = 0.0,
        singles_penalty_max: float = 1.85,
        singles_range: float = 0.95,
        fom_method: str = 'angle',
        **FOM_kwargs) -> float:
    """
    # General call for the figure of merit (FOM) of a permuted cluster

    The FOM method takes in an ordered cluster of interactions from the supplied
    event and returns the FOM using the specified method. Argonne Forward
    Tracking (AFT) uses a FOM that compares angles computed from the recorded
    positions of interactions and an expected scattering angle calculated using
    the Compton Scattering Formula (CSF). Orsay Forward Tracking (OFT) uses the
    geometric scattering to compute an expected scattering energy and typically
    returns a probability of the error and of the likelihood of attenuation.

    ## Arguments
    - `event` : gamma-ray event object containing interactions
    - `permutation` : permutation containing ordered interaction indices
    - `start_point` : often the origin, where the gamma-ray originated from
    - `start_energy` : starting energy for the gamma-ray, often the total energy
    of the cluster, but can be set independently (e.g., partial FOM, TANGO
    estimate)
    - `estimate_start_energy` : use the TANGO estimate
    - `normalize_start_energy_estimate` : use the sigma normalized TANGO estimate
    - `eres` : error in energy measurement (MeV)
    - `Nmi` : number of interactions in the cluster, often the total number in
    the permutation, but can be set independently (e.g., partial FOM)
    - `singles_method` : method for handling singles
    - `singles_penalty_min` : minimum value of singles method
    - `singles_penalty_max` : maximum value of singles method
    - `fom_method` : programmed FOM methods

    ## Singles methods
    - `"continuous"` : assigns a continuous penalty based on distance and
    cross-section
    - `None` : return zero
    - `"range"` : assigns singles outside of an attenuation range a fixed penalty
    - `"probability"` : returns a weighted probability, similar to continuous
    - `"depth"` : returns a range measure specified by typical chat file

    ## FOM methods
    - `"agata" | "oft"` : uses the scattered energy
    - `"angle | "aft"` (default) : uses the angle of scattering
    - `"cosine"` : uses the cosine of the angle of scattering
    - `"geo_local"` : uses scattered energy and local energy estimate
    - `"local"` : uses the local energy estimate
    - `"tango_variance"` : uses the variance of TANGO energy estimates
    (completely energy sum independent but inaccurate)
    - `"feature"` : uses a multitude of FOM features and weights to compute
    - `"selected"` : uses a selected subset of FOM features and weights to compute

    ## Returns
    - `FOM` : the figure of merit specified by the `*args` and `**kwargs`
    """
    perm = tuple(permutation)
    if start_point in perm and perm[0] == start_point:
        perm = perm[1:]
    elif start_point in perm:
        print('The start index is in the permutation:')
        print(f' - Permutation {perm}')
        print(f' - Start index {start_point}')
        raise IndexError

    if len(perm) == 1:
        return single_FOM(event, permutation=perm,
                          start_point=start_point,
                          singles_method=singles_method,
                          singles_penalty_min=singles_penalty_min,
                          singles_penalty_max=singles_penalty_max,
                          singles_range=singles_range)

    # if monster_size is not None:
    #     if len(perm) >= monster_size:
    #         return monster_penalty

    if start_energy is None:
        start_energy = sum(event.points[i].e for i in perm)
    if estimate_start_energy and len(perm) > 2: # estimate makes FOM = 0 if len is 1
        if normalize_start_energy_estimate:
            estimate = event.estimate_start_energy_sigma_weighted_perm(perm,
                                                                       start_point=start_point,
                                                                       eres=eres)
        else:
            estimate = event.estimate_start_energy_perm(perm, start_point=start_point)
        if accept_max:
            if start_energy + min_excess < estimate < start_energy + max_excess:
                start_energy = estimate
        else:
            start_energy = estimate

    fom_method = fom_method.lower()
    # if fom_method == 'agata':
    #     return agata_FOM(event,perm,
    #                         start_energy=start_energy,
    #                         start_point=start_point,
    #                         Nmi=Nmi,
    #                         **FOM_kwargs)
    # if fom_method == 'angle' or fom_method == 'aft':
    #     return angle_FOM(event,perm,
    #                         start_energy=start_energy,
    #                         start_point=start_point,
    #                         Nmi=Nmi,
    #                         **FOM_kwargs)
    # if fom_method == 'agata_exp' or fom_method == 'oft':
    #     FOM_kwargs['exponential'] = True
    #     return agata_FOM(event,perm,
    #                         start_energy=start_energy,
    #                         start_point=start_point,
    #                         Nmi=Nmi,
    #                         **FOM_kwargs)
    # if fom_method == 'cos':
    #     return cosine_FOM(event,perm,
    #                           start_energy=start_energy,
    #                           start_point=start_point,
    #                           Nmi=Nmi,
    #                           **FOM_kwargs)
    # if fom_method == 'local':
    #     return local_tango_FOM(event,perm,
    #                                start_energy=start_energy,
    #                                start_point=start_point,
    #                                min_excess = min_excess,
    #                                max_excess = max_excess,
    #                                Nmi=Nmi,
    #                                **FOM_kwargs)
    # if fom_method == 'geo_loc' or from_method ==  'geo_local':
    #     return geo_loc_FOM(event,perm,
    #                            start_energy=start_energy,
    #                            start_point=start_point,
    #                            Nmi=Nmi,
    #                            **FOM_kwargs)
    # if fom_method == 'tango_variance':
    #     return tango_variance(event,perm,
    #                               start_point=start_point)
    # if fom_method == 'feature':
    #     return feature_FOM(event, perm,
    #                            start_energy=start_energy,
    #                            start_point=start_point,
    #                            Nmi=Nmi,
    #                            eres=eres,
    #                            **FOM_kwargs)
    # if fom_method == 'selected':
    #     return selected_FOM(event, perm,
    #                             start_point=start_point,
    #                             start_energy=start_energy,
    #                             Nmi=Nmi,
    #                             eres=eres,
    #                             **FOM_kwargs)
    # if fom_method == 'reduce' or fom_method == 'transition':
    #     return reduction_FOM(event, perm, start_point=start_point,
    #                              **FOM_kwargs)
    # raise NotImplementedError('The FOM method name is not implemented.')
    match fom_method:
        case 'agata':
            return agata_FOM(event,perm,
                            start_energy=start_energy,
                            start_point=start_point,
                            Nmi=Nmi,
                            **FOM_kwargs)
        case 'angle' | 'aft':
            return angle_FOM(event,perm,
                            start_energy=start_energy,
                            start_point=start_point,
                            Nmi=Nmi,
                            **FOM_kwargs)
        case 'agata_exp' | 'oft':
            FOM_kwargs['exponential'] = True
            return agata_FOM(event,perm,
                            start_energy=start_energy,
                            start_point=start_point,
                            Nmi=Nmi,
                            **FOM_kwargs)
        case 'cos':
            return cosine_FOM(event,perm,
                              start_energy=start_energy,
                              start_point=start_point,
                              Nmi=Nmi,
                              **FOM_kwargs)
        case 'local':
            return local_tango_FOM(event,perm,
                                   start_energy=start_energy,
                                   start_point=start_point,
                                   min_excess = min_excess,
                                   max_excess = max_excess,
                                   Nmi=Nmi,
                                   **FOM_kwargs)
        case 'geo_loc' | 'geo_local':
            return geo_loc_FOM(event,perm,
                               start_energy=start_energy,
                               start_point=start_point,
                               Nmi=Nmi,
                               **FOM_kwargs)
        case 'tango_variance':
            return tango_variance(event,perm,
                                  start_point=start_point)
        case 'feature':
            return feature_FOM(event, perm,
                               start_energy=start_energy,
                               start_point=start_point,
                               Nmi=Nmi,
                               eres=eres,
                               **FOM_kwargs)
        case 'selected':
            return selected_FOM(event, perm,
                                start_point=start_point,
                                start_energy=start_energy,
                                Nmi=Nmi,
                                eres=eres,
                                **FOM_kwargs)
        case 'reduce' | 'transition':
            return reduction_FOM(event, perm, start_point=start_point,
                                 **FOM_kwargs)
        case _ :
            raise NotImplementedError('The FOM method name is not implemented.')

#%% Predefined FOMs

def agata_FOM(event: Event, permutation: Iterable[int],
              start_point:int = 0,
              start_energy:float = None,
              Nmi:int = None,
              eres:float = 1e-3,
              exponential:bool = False,
              debug=False,
              **kwargs) -> float:
    """
    # AGATA or OFT FOM

    This function calculates the figure of merit (FOM) for a given event and a
    permutation of its interactions. The FOM is a measure of how well the event
    matches the expected physics model of gamma-ray tracking. The FOM is used
    for validation and selection of the best permutation for tracking. The FOM
    is based on the AGATA or OFT FOM, which is part of the Orsay Forward
    Tracking Algorithm (OFT).

    ## Args:
    - `event` : An Event object containing the interactions in the cluster being
      evaluated
    - `permutation` : An iterable of integers that represents a permutation of
      the interactions in a cluster from the event
    - `start_point` (optional) : An integer that indicates the index of the
      interaction point to start from in the event. The default value is 0,
      which means the first interaction point is the origin.
    - `start_energy` (optional) : A float that indicates the initial energy of
      the &gamma;-ray in MeV. The default value is None, which means the sum of
      the energies of all the interactions in the permutation.
    - `Nmi` (optional) : ("N minus i") An integer that indicates the number of
      interactions in the full permutation. The default value is None, which
      means the length of the permutation. In cases where only part of the
      permutation is presented to the FOM function, this should be specified to
      ensure accurate calculation of uncertainties when additional energies
      outside of the permutation are used.
    - `eres` (optional) : A float that indicates the energy resolution of the
      detector in MeV. The default value is 1e-3, which means 1 keV.
    - `exponential` (optional) : A boolean that indicates whether to return the
      exponential of the FOM or not. The default value is False, which means the
      FOM is returned as it is. The OFT algorithm uses the exponential of the
      value returned in this default case. The negative log of the typical OFT
      value is used here instead to allow linear combination of elements and to
      align with the goal of minimization (like other FOMs).
    - `debug` (optional) : A boolean that indicates whether to print debug
      messages to the standard output or not. The default value is False, which
      means no debug messages are printed.
    - `**kwargs` : Keyword arguments (currently unused)

    ## Returns:
    - A float that represents the FOM value for the given event and permutation

    TODO - The original OFT includes many controls for clustering by tuning &alpha;
    """
    if len(permutation) <= 1:
        return None
    if start_energy is None:
        start_energy = np.sum(event.energy_matrix[list(permutation)])
    if Nmi is None:
        Nmi = len(permutation)

    r_sum_geo = np.abs(event.res_sum_geo(permutation,
                                         start_point=start_point,
                                         start_energy=start_energy))
    r_sum_geo_v = r_sum_geo / np.abs(event.res_sum_geo_sigma(permutation,
                                                             start_point=start_point,
                                                             start_energy=start_energy,
                                                             Nmi=Nmi + 1,
                                                             eres=eres))

    # The accuracy of the first angle measure is increased by the increased
    # accuracy in the position of the origin
    r_sum_geo_v[0] *= np.sqrt(2)

    cross_abs = event.linear_attenuation_abs(permutation, start_point=start_point,
                                             start_energy=start_energy)
    cross_compt = event.linear_attenuation_compt(permutation, start_point=start_point,
                                                 start_energy=start_energy)
    cross_pair = event.linear_attenuation_pair(permutation, start_point=start_point,
                                               start_energy=start_energy)

    cross_total = cross_abs + cross_compt + cross_pair

    ge_distances = event.ge_distance_perm(permutation, start_point=start_point)

    if debug:
        esums = event.cumulative_energies(permutation, start_point, start_energy)
        egeos = event.scattered_energy(permutation, start_point, start_energy)
        errors = event.res_sum_geo_sigma(permutation,
                                        start_point=start_point,
                                        start_energy=start_energy,
                                        Nmi=Nmi + 1,
                                        eres=eres)
        err_cos = event.cos_act_err_perm(permutation=permutation,
                                            start_point=start_point)

        for ii, (i,j,k) in enumerate(zip([start_point] + list(permutation),
                                         permutation,
                                         permutation[1:])):
            print(f'{ii} : {i}, {j}, {k}:')
            print(f'   err_cos : {err_cos[ii]:>4.7f}  eres : {eres:>4.7f}')
            print(f' escattern : {egeos[ii]:>4.7f}')
            print(f'  escatter : {esums[ii+1]:>4.7f}')
            print(f'  residual : {abs(esums[ii+1] - egeos[ii]):>4.7f}')
            print(f'     error : {errors[ii]**2:>4.7f}')
            print(f'prob_error : {abs(esums[ii+1] - egeos[ii])**2/errors[ii]**2:>4.7f}')
            print(f'prob_error : {r_sum_geo_v[ii]**2:>4.7f}')
            print(f'prob_compt : {-np.log(cross_compt[ii]/cross_total[ii]):>4.7f}')
            print(f'prob_distance : {cross_compt[ii]*ge_distances[ii]:>4.7f}')
        print(f'{permutation[-2]}, {permutation[-1]}:')
        print(f'prob_abs : {-np.log(cross_abs[-1]/cross_total[-1])}')
        print(f'prob_distance : {cross_abs[-1]*ge_distances[-1]}')

    out = (
           np.sum(r_sum_geo_v**2) + \
           np.sum(cross_compt[:-1]*ge_distances[:-1] - \
                  np.log(cross_compt[:-1]/cross_total[:-1])) + \
           cross_abs[-1]*ge_distances[-1] - np.log(cross_abs[-1]/cross_total[-1])\
          ) \
          /(2*len(permutation) - 1)

    if exponential:
        return np.exp(-out)
    return out

def angle_FOM(event: Event, permutation: Iterable[int],
              start_point:int = 0,
              start_energy:float = None,
              Nmi:int = None,
              eres:float = 1e-3,
              penalty:float = 0.4,
              **kwargs) -> float:
    """Angle FOM used with GRETINA"""
    if len(permutation) <= 1:
        return None
    if start_energy is None:
        start_energy = np.sum(event.energy_matrix[list(permutation)])
    if Nmi is None:
        Nmi = len(permutation)

    # r_theta = np.abs(event.res_theta(permutation,
    #                                  start_point=start_point,
    #                                  start_energy=start_energy))

    # r_theta_v = r_theta / np.abs(event.res_theta_sigma(permutation,
    #                                                    start_point=start_point,
    #                                                    start_energy=start_energy,
    #                                                    Nmi=Nmi,
    #                                                    eres=eres))

    r_theta_cap = np.abs(event.res_theta_cap(permutation,
                                     start_point=start_point,
                                     start_energy=start_energy))

    # r_theta_cap_v = r_theta_cap / np.abs(event.res_theta_sigma(permutation,
    #                                                    start_point=start_point,
    #                                                    start_energy=start_energy,
    #                                                    Nmi=Nmi,
    #                                                    eres=eres))

    comp_penalty = np.abs(event.compton_penalty_ell1(permutation,
                                                     start_point=start_point,
                                                     start_energy=start_energy))

    return np.sqrt(np.sum(r_theta_cap**2) + np.sum(penalty * comp_penalty))/(Nmi - 1)

def cosine_FOM(event: Event, permutation: Iterable[int],
              start_point:int = 0,
              start_energy:float = None,
              Nmi:int = None,
              eres:float = 1e-3,
              penalty:float = 0.4,
              **kwargs) -> float:
    """Cosine of angle FOM used with GRETINA"""
    if len(permutation) <= 1:
        return None
    if start_energy is None:
        start_energy = np.sum(event.energy_matrix[list(permutation)])
    if Nmi is None:
        Nmi = len(permutation)

    # r_cosines = np.abs(event.res_cos(permutation,
    #                                  start_point=start_point,
    #                                  start_energy=start_energy))

    # r_cosines_v = r_cosines / np.abs(event.res_cos_sigma(permutation,
    #                                                      start_point=start_point,
    #                                                      start_energy=start_energy,
    #                                                      Nmi=Nmi,
    #                                                      eres=eres))

    r_cosines_cap = np.abs(event.res_cos_cap(permutation,
                                             start_point=start_point,
                                             start_energy=start_energy))

    # r_cosines_cap_v = r_cosines_cap / np.abs(event.res_cos_sigma(permutation,
    #                                                              start_point=start_point,
    #                                                              start_energy=start_energy,
    #                                                              Nmi=Nmi,
    #                                                              eres=eres))

    comp_penalty = np.abs(event.compton_penalty(permutation,
                                                start_point=start_point,
                                                start_energy=start_energy))

    return np.sqrt(np.sum(r_cosines_cap**2))/(Nmi - 1) + penalty * np.sum(comp_penalty)
    # raise NotImplementedError

def local_tango_FOM(event: Event, permutation: Iterable[int],
              start_point:int = 0,
              start_energy:float = None,
              Nmi:int = None,
              eres:float = 1e-3,
              use_variance:bool = False,
              norm:int = 2,
              **kwargs):
    """FOM using local energy estimates"""
    r_sum_loc = np.abs(event.res_sum_loc(permutation,
                                         start_point=start_point,
                                         start_energy=start_energy))
    if use_variance:
        r_sum_loc_v = r_sum_loc / np.abs(event.res_sum_loc_sigma(permutation,
                                                             start_point=start_point,
                                                             Nmi=Nmi,
                                                             eres=eres))
        return np.sum(r_sum_loc_v**norm)
    return np.sum(r_sum_loc**norm)

def geo_loc_FOM(*args, **kwargs):
    """FOM using scattered energy and local energy estimate"""
    # TODO
    raise NotImplementedError

def tango_variance(*args, **kwargs):
    """FOM using the variance of TANGO estimates"""
    # TODO
    raise NotImplementedError

def feature_FOM(event: Event,
                permutation: Iterable[int],
                weights: np.ndarray[float] = None,
                start_point:int = 0,
                start_energy:float = None,
                Nmi:int = None,
                eres:float = 1e-3) -> float:
    """FOM derived from the full set of FOM features"""
    if weights is None:
        raise ValueError
    a = FOM_features(event, permutation, start_point=start_point,
                     start_energy=start_energy,
                     Nmi=Nmi, eres=eres)
    # TANGO estimate
    e_tango_var = event.estimate_start_energy_sigma_weighted_perm(permutation)
    e_sum = sum(event.points[i].e for i in permutation)
    start_e = np.max([e_tango_var, e_sum])
    # features with a TANGO estimate (if we make one)
    if start_e > e_sum:
        b = FOM_features(event, permutation, start_point = start_point,
                         start_energy = start_e,
                         Nmi=Nmi, eres=eres)
    else:
        b = a
    out = np.array(list(a.values()) + list(b.values()))
    return np.dot(weights, out)

#%% Singles treatments

def single_FOM(event: Event, permutation: Iterable[int],
               start_point: int = 0,
               singles_method: str = 'depth',
               singles_penalty_min: float = 0.0,
               singles_penalty_max: float = 1.85,
               singles_range: float = 0.82,
               small_depth: float = 0.59) -> float:
    """
    # FOM for single interaction &gamma;-rays

    This function calculates the figure of merit (FOM) for a single interaction
    &gamma;-ray. The FOM for a single interaction &gamma;-ray cannot be
    calculated using the Compton Scattering Formula, so it requires special
    treatment based on energy and position only.

    ## Args:
    - `event` : An Event object that represents the event to be evaluated
    - `permutation` : An iterable of integers that represents a permutation of
      the interactions in the event. For a single interaction event, this should
      be an iterable with one element.
    - `start_point` (optional) : An integer that indicates the index of the
      interaction point to start from in the permutation. The default value is
      `0`, which means the detector origin.
    - `singles_method` (optional) : A string that indicates the method to use
      for calculating the FOM for a single interaction &gamma;-ray. The default
      value is `'depth'`, which means an interpolated version of the range
      method with additional acceptance in the very short range region. Other
      possible values are:
        - `"continuous"` : assigns a continuous penalty based on distance and
          linear attenuation
        - `None` : returns zero
        - `"range"` : assigns singles outside of an attenuation range a fixed
          penalty
        - `"probability"` : returns a weighted probability, similar to
          continuous
        - `"depth"` : returns a range measure specified by typical chat file
    - `singles_penalty_min` (optional) : A float that indicates the minimum
      value of the FOM. The default value is 0.0.
    - `singles_penalty_max` (optional) : A float that indicates the scale factor
      for the FOM, or the maximum value for indicator FOMs. The default value is
      1.85.
    - `singles_range` (optional) : A float that indicates the longest range that
      is acceptable for an indicator FOM. The default value is 0.82 to best
      match the interpolated `'depth'` method.
    - `small_depth` (optional) : A float that allows the acceptance of all
      interactions within some small depth into the crystal. The default value
      is 0.59 cm.

    ## Returns:
    - A float that represents the FOM value for the single interaction
    &gamma;-ray.
    """
    if len(permutation) != 1:
        raise ValueError("The length of the permutation is more than a single"+\
            " interaction and a singles method should not be used")
    singles_method = singles_method.lower()
    # if singles_method is None:
    #     return 0.
    # if singles_method == "continuous":
    #     return singles_continuous(event, permutation, start_point=start_point,
    #                                   singles_penalty_min=singles_penalty_min,
    #                                   singles_penalty_max=singles_penalty_max)
    # if singles_method == "proba" or singles_method == "probability":
    #     return singles_proba(event, permutation, start_point=start_point,
    #                              singles_penalty_min=singles_penalty_min,
    #                              singles_penalty_max=singles_penalty_max)
    # if singles_method == "range":
    #     return singles_in_range(event, permutation, start_point=start_point,
    #                                 singles_penalty_min=singles_penalty_min,
    #                                 singles_penalty_max=singles_penalty_max,
    #                                 singles_range=singles_range)
    # if singles_method == "depth" or singles_method == "chat":
    #     return singles_depth(event, permutation, start_point=start_point,
    #                              singles_penalty_min=singles_penalty_min,
    #                              singles_penalty_max=singles_penalty_max,
    #                              singles_range=singles_range)
    # raise NotImplementedError("The singles method chosen is not implemented.")
    match singles_method:
        case None:
            return 0.
        case "continuous":
            return singles_continuous(event, permutation, start_point=start_point,
                                      singles_penalty_min=singles_penalty_min,
                                      singles_penalty_max=singles_penalty_max,
                                      small_depth=small_depth)
        case "proba" | "probability":
            return singles_proba(event, permutation, start_point=start_point,
                                 singles_penalty_min=singles_penalty_min,
                                 singles_penalty_max=singles_penalty_max,
                                 small_depth=small_depth)
        case "range":
            return singles_in_range(event, permutation, start_point=start_point,
                                    singles_penalty_min=singles_penalty_min,
                                    singles_penalty_max=singles_penalty_max,
                                    singles_range=singles_range,
                                    small_depth=small_depth)
        case "depth" | "chat":
            return singles_depth(event, permutation, start_point=start_point,
                                 singles_penalty_min=singles_penalty_min,
                                 singles_penalty_max=singles_penalty_max,
                                 singles_range=singles_range)
        case _:
            raise NotImplementedError("The singles method chosen is not implemented.")

def singles_continuous(event: Event, permutation: Iterable[int],
                       start_point: int = 0,
                       singles_penalty_min: float = 0.0,
                       singles_penalty_max: float = 1.85,
                       small_depth: float = 0.59) -> float:
    """
    # Assign a continuous penalty based on distance and linear attenuation

    ## Args:
    - `event` : An Event object that represents the event to be evaluated
    - `permutation` : An iterable of integers that represents a permutation of
      the interactions in the event. For a single interaction event, this should
      be an iterable with one element.
    - `start_point` (optional) : An integer that indicates the index of the
      interaction point to start from in the permutation. The default value is
      `0`, which means the detector origin.
    - `singles_penalty_min` (optional) : A float that indicates the minimum
      value of the FOM. The default value is 0.0.
    - `singles_penalty_max` (optional) : A float that indicates the scale factor
      for the FOM, or the maximum value for indicator FOMs. The default value is
      1.85.
    - `small_depth` (optional) : A float that allows the acceptance of all
      interactions within some small depth into the crystal. The default value
      is 0.59 cm.

    ## Returns:
    - A float that represents the FOM value for the single interaction
    &gamma;-ray.
    """
    distance = event.ge_distance[start_point, permutation[0]]
    if distance < small_depth:
        distance = 0
    linear_attenuation = event.lin_mu_total(permutation=permutation,
                                            start_point=start_point)[0]
    return singles_penalty_max * linear_attenuation * distance + singles_penalty_min

def singles_proba(event: Event, permutation: Iterable[int],
                  start_point: int = 0,
                  singles_penalty_min: float = 0.0,
                  singles_penalty_max: float = 1.85,
                  small_depth: float = 0.59) -> float:
    """
    # Cumulative probability of the ray traveling further than the distance

    ## Args:
    - `event` : An Event object that represents the event to be evaluated
    - `permutation` : An iterable of integers that represents a permutation of
      the interactions in the event. For a single interaction event, this should
      be an iterable with one element.
    - `start_point` (optional) : An integer that indicates the index of the
      interaction point to start from in the permutation. The default value is
      `0`, which means the detector origin.
    - `singles_penalty_min` (optional) : A float that indicates the minimum
      value of the FOM. The default value is 0.0.
    - `singles_penalty_max` (optional) : A float that indicates the scale factor
      for the FOM, or the maximum value for indicator FOMs. The default value is
      1.85.
    - `small_depth` (optional) : A float that allows the acceptance of all
      interactions within some small depth into the crystal. The default value
      is 0.59 cm.

    ## Returns:
    - A float that represents the FOM value for the single interaction
    &gamma;-ray.
    """
    return singles_penalty_max * \
        np.exp(-singles_continuous(event,permutation,
                                   start_point=start_point,
                                   singles_penalty_max=1,
                                   singles_penalty_min=0,
                                   small_depth=small_depth)) + \
        singles_penalty_min

def singles_in_range(event: Event, permutation: Iterable[int],
                  start_point: int = 0,
                  singles_penalty_min: float = 0.0,
                  singles_penalty_max: float = 1.85,
                  singles_range: float = 0.82,
                  small_depth: float = 0.59) -> float:
    """
    # Indicator if the interaction is in range or out of range

    ## Args:
    - `event` : An Event object that represents the event to be evaluated
    - `permutation` : An iterable of integers that represents a permutation of
      the interactions in the event. For a single interaction event, this should
      be an iterable with one element.
    - `start_point` (optional) : An integer that indicates the index of the
      interaction point to start from in the permutation. The default value is
      `0`, which means the detector origin.
    - `singles_penalty_min` (optional) : A float that indicates the minimum
      value of the FOM. The default value is 0.0.
    - `singles_penalty_max` (optional) : A float that indicates the scale factor
      for the FOM, or the maximum value for indicator FOMs. The default value is
      1.85.
    - `singles_range` (optional) : A float that indicates the longest range that
      is acceptable for an indicator FOM. The default value is 0.82 to best
      match the interpolated `'depth'` method.
    - `small_depth` (optional) : A float that allows the acceptance of all
      interactions within some small depth into the crystal. The default value
      is 0.59 cm.

    ## Returns:
    - A float that represents the FOM value for the single interaction
    &gamma;-ray.
    """
    in_range = singles_proba(event, permutation,
                             start_point=start_point,
                             singles_penalty_min=0.,
                             singles_penalty_max=1.,
                             small_depth=small_depth) > (singles_range)
    return singles_penalty_max * in_range + singles_penalty_min * (1 - in_range)

def singles_depth(event: Event, permutation: Iterable[int],
                  singles_penalty_min: float = 0.0,
                  singles_penalty_max: float = 1.85,
                  detector:DetectorConfig=default_config,
                  **kwargs) -> float:
    """
    # Reject singles based on their energy and depth

    This is the default used in GRETINA tracking code. Approximately rejection
    at 81%-83% or gamma-ray range. Data is from AFT tracking chat file and
    indicates energy [MeV] and depth [cm].

    ## Args:
    - `event` : An Event object that represents the event to be evaluated
    - `permutation` : An iterable of integers that represents a permutation of
      the interactions in the event. For a single interaction event, this should
      be an iterable with one element.
    - `singles_penalty_min` (optional) : A float that indicates the minimum
      value of the FOM. The default value is 0.0.
    - `singles_penalty_max` (optional) : A float that indicates the scale factor
      for the FOM, or the maximum value for indicator FOMs. The default value is
      1.85.

    ## Returns:
    - A float that represents the FOM value for the single interaction
    &gamma;-ray.
    """
    assert len(permutation) == 1
    interaction = event.points[permutation[0]]
    depth = np.linalg.norm(interaction.x) - detector.get_inner_radius()
    energy = interaction.e
    #                Energy [MeV],    Depth [cm]
    data = np.array([[0.000,          0.59],
                     [0.080,          0.59],
                     [0.100,          0.62],
                     [0.150,          1.38],
                     [0.200,          2.07],
                     [0.300,          3.05],
                     [0.400,          3.69],
                     [0.500,          4.19],
                     [0.600,          4.62],
                     [0.800,          5.36],
                     [1.000,          6.01],
                     [1.250,          6.75],
                     [1.500,          7.40],
                     [2.000,          8.43],
                     [3.000,          9.77],
                     [4.000,         10.52],
                     [5.000,         10.91],
                     [16.383,        27.41]])
    if depth > np.interp(energy, data[:,0], data[:,1]):
        return singles_penalty_max
    return singles_penalty_min

#%% Ordering routines

# TODO - cleanup

def semi_greedy_clusters(event: Event, clusters: Dict[Hashable,Iterable],
                         width:int = 3,
                         stride:int = 1,
                         direction:str = 'forward',
                         **FOM_kwargs):
    """
    Apply a semi-greedy clustering to all of the clusters.
    """
    best_ordered_clusters = {s: semi_greedy(event,
                                            list(cluster),
                                            width=width,
                                            stride=stride,
                                            direction=direction,
                                            **FOM_kwargs)
                                for (s, cluster) in clusters.items()}
    return best_ordered_clusters

def semi_greedy(event: Event, cluster:Iterable=None,
                width:int = 3,
                stride:int = 1,
                direction:str = 'forward',
                return_score:bool = False,
                max_cluster_size:int = 8,
                short_stop:bool = False,
                debug:bool = False,
                **FOM_kwargs) -> List:
    """
    # Use a semi-greedy approach to find the optimal order of a single cluster

    For the forward direction, we assume that the cluster contains a complete
    &gamma;-ray, and we attempt to find chunks of permutations that optimize
    that part of the total permutation. The `width` parameter controls how many
    points are evaluated for comparison (given n interactions, permutations of
    length `width` are created, evaluated, and the optimal w.r.t. FOM is
    selected). The `stride` parameter controls how many interactions from the
    chosen optimal permutation of size `width` are frozen for the next step of
    the algorithm. The method will then search for the best permutation of the
    next `width` interactions.

    If `width` is greater than the length of the cluster, the method performs a
    complete enumeration. The most greedy possible value for `width` is `2`
    (this is the minimum required for all FOMs that use the Compton Scattering
    Formula).

    Backward tracking involves ordering and clustering such that there is no
    assumption about complete &gamma;-rays. Backward tracking is described in
    more detail in the corresponding function `semi_greedy_backward`.

    Args:
        - `event`: the &gamma;-ray Event object
        - `cluster` : the indices of the cluster to be ordered
        - `width` : the width of the combinatorial window
        - `stride` : the number of accepted points from each window
        - `direction` : the direction of the greedy approach
        - `return_score` : include the FOM value in the output
        - `max_cluster_size` : do not sort clusters with more interactions than this
        - `short_stop` : stop after the first stride, only the first interactions are important
        - `debug`: print debug output
        - `**FOM_kwargs` : specifications for the type of FOM to use
    """
    # FOM_kwargs['filter_singles'] = False
    if cluster is None:
        cluster = list(range(0,len(event.hit_points) + 1))
    if FOM_kwargs.get('fom_method') == 'agata':
        FOM_kwargs['negative'] = True
    if len(cluster) > max_cluster_size:
        if return_score:
            return (tuple(cluster), FOM(event, cluster,**FOM_kwargs))
        return cluster
    if direction=='forward':
        curr_e = sum(event.points[i].e for i in cluster)
        # excess_e = 0
        start_point = 0
        order = []
        remaining_points = list(deepcopy(cluster))
        best_perm = [0]*width
        while len(order) < len(cluster):
            # start_point_index = best_perm[stride - 1]
            best_perm = remaining_points[:width]
            best_score = np.inf
            # Special case where excluding the last point does not result in a
            # reduced number of evaluations, just removes last point from computation
            if len(remaining_points) == width + 1:
                width = len(remaining_points)
            for perm in permutations(remaining_points, r=min(width, len(remaining_points))):
                score = FOM(event, order + list(perm), start_energy=curr_e,
                                #  excess_e= excess_e,
                                    start_point=start_point, **FOM_kwargs)
                # print(f'  {order + list(perm), score}')
                if score < best_score:
                    if debug:
                        print(f'***{order + list(perm), score}')
                    best_perm = perm
                    best_score = score
            # print('**************')
            # Can accept all remaining points if they are all included in the combinatorial window
            if width >= len(remaining_points):
                order.extend(best_perm)
            else:
                order.extend(best_perm[:stride])
                # print(f'{order} {best_score}')
                for point in best_perm[:stride]:
                    # excess_e += self.points[point].e
                    remaining_points.remove(point)
                    # curr_e -= self.points[point].e
            if short_stop:
                return order
        if return_score:
            # print(FOM_kwargs)
            # print(self.FOM(order,**FOM_kwargs))
            return (tuple(order), FOM(event, order, **FOM_kwargs))
        return order
    if direction=='backward':
        return semi_greedy_backward(event, cluster, width=width,
                                    stride=stride, debug=debug,
                                    **FOM_kwargs)
    if direction=='hybrid':
        raise NotImplementedError

def semi_greedy_backward(event: Event, subset:Iterable = None,
                         width:int=3, stride:int=1,
                         e_min:float=0.09, e_max:float=0.25,
                         debug:bool=False,
                         validation_FOM_kwargs:Dict= None,
                         **FOM_kwargs) -> Dict[int,Iterable[int]]:
    """
    Order and cluster a subset of interactions in an event using a semi-greedy
    backward tracking approach.

    This method is based on the semi-greedy forward tracking method, but it
    starts from the end of the subset and moves backwards. It explores
    permutations of interactions with a given width and accepts them with a
    given stride. It uses a figure of merit (FOM) to evaluate the quality of the
    ordering and clustering. This is not the same algorithm used in the original
    backtracking method by van der Marel and Cederwall [1] that uses geometry as
    a filter for interactions.

    Parameters
    ----------
    event : Event
        The &gamma;-ray Event object.
    subset : Iterable, optional
        The indices of the interactions to be ordered and clustered. If None,
        use all interactions in the event. Default is None.
    width : int, optional
        The number of interactions considered for the combinatorial window. Default is 3.
    stride : int, optional
        The number of interactions accepted from the combinatorial window. Default is 1.
    e_min : float, optional
        The minimum energy for designating a startable interaction. Default is 0.09.
    e_max : float, optional
        The maximum energy for designating a startable interaction. Default is 0.25.
    debug : bool, optional
        Whether to print debug output or not. Default is False.
    validation_FOM_kwargs : Dict, optional
        The keyword arguments for the FOM used for cluster validation. Default is None.
    **FOM_kwargs
        The keyword arguments for the FOM used for ordering.

    Returns
    -------
    dict of clusters
        A dict of clusters (each cluster is a list of interaction indices in the
        order they were accepted) where the cluster with key -1 contains unassigned points.

    References
    ----------
    [1] H. van der Marel and T. Cederwall, "A new method for ordering and
    clustering &gamma;-ray data", Nuclear Instruments and Methods in Physics
    Research Section A: Accelerators, Spectrometers, Detectors and Associated
    Equipment 423 (1999) 468-479.
    (https://doi.org/10.1016/S0168-9002(99)00801-3)
    """
    if subset is None:
        subset = list(range(0,len(event.hit_points) + 1))
    # i2e = {i : event.points[i].e for i in cluster} # conversion from index to energy
    # e2i = {event.points[i].e : i for i in subset} # conversion from energy to index

    # sort energies in ascending order (want to start process with smallest energy)
    es = sorted([ event.points[i].e for i in subset ])
    # es = sorted(es)
    if debug:
        print(es)
    # print(i2e, e2i)
    # Get the points with startable energies for back-tracking
    startable = [(e_min < e < e_max) for e in es]
    if debug:
        print(dict(zip(subset, startable)))
    records = {}
    # Loop over the possible starting points
    for i, start in enumerate(startable):
        if start:
            # Initialize pool of backtrack-able points (remaining_points), FOM
            # (best_score), and current perm (chosen_points)
            # start_index = e2i[es[i]]
            start_index = subset[i]
            remaining_points = list(subset)
            if 0 not in remaining_points: # Must include origin for backtracking
                remaining_points += [0]
            remaining_points.remove(start_index)
            best_score = np.inf
            improving = True
            chosen_points = [start_index]
            if debug:
                print(f'Starting with index {start_index}')
            while improving:
                improving = False
                best_perm = []
                if debug:
                    print('Loop')
                for w in range(1, width): # permutations starting with 0
                    for perm in permutations(remaining_points, r=min(w, len(remaining_points))):
                        print(f'   Trying perm: {perm} + {chosen_points}')
                        if (0 in perm and perm[0] != 0) :
                            if debug:
                                print(f'   Skipping perm {perm}, bad origin')
                            continue
                        if (perm[0] == 0 and len(perm) + len(chosen_points) == 2):
                            if debug:
                                print(f'   Skipping perm {perm}, single')
                            continue
                        try:
                            score = FOM(event, tuple(list(perm[:]) + chosen_points),
                                        start_point=perm[0],
                                        **FOM_kwargs)
                        except Exception as exc:
                            print('Encountered a problem when evaluating the FOM for perm:'+\
                                f'{perm} and chosen_points: {chosen_points}')
                            raise Exception from exc
                        if score <= best_score:
                            best_perm = perm
                            best_score = score
                            if debug:
                                print('***')
                                print(perm, chosen_points)
                                print(score)
                            improving = True
                if debug:
                    print(f'Best perm: {best_perm}, Remaining points: {remaining_points}')
                for point in best_perm[-stride:]:
                    remaining_points.remove(point)
                chosen_points = list(best_perm)[-stride:] + chosen_points
                if debug:
                    print(f'New chosen points {chosen_points}')
                best_score = FOM(event, chosen_points[:],
                                       start_point=chosen_points[0],
                                       **FOM_kwargs)
                if debug:
                    print(f'Best perm: {best_perm},' + \
                          f' Remaining points: {remaining_points},' + \
                          f' Improving? {improving}')
                if chosen_points[0] == 0:
                    break
            records[tuple(chosen_points)] = (best_score, event.energy_sum(chosen_points))
            if debug:
                print(f'Added record: {records}')

    if debug:
        print('Order records')
        print(records)

    if validation_FOM_kwargs is not None:
        new_records = {}
        for record_id, record in records.items():
            new_records[record_id] = (FOM(event, record_id, **validation_FOM_kwargs), record[1])
        records = new_records
        if debug:
            print('Validation records')
            print(records)

    removed_points = set()
    clusters = {}
    i = 0
    while len(records) > 0:
        best_score = np.inf
        best_cluster = []
        # best_cluster_id = None
        for record_id, record in records.items():
            if record[0] < best_score:
                best_cluster = record_id
                if best_cluster[0] == 0:
                    best_cluster = best_cluster[1:]
                best_score = record[0]
        removed_points = removed_points.union(best_cluster)
        if debug:
            print(f'Selected the best cluster {best_cluster}')
        clusters[i] = list(best_cluster)
        i += 1
        removal_indices = []
        for record_id, record in records.items():
            if any(index in removed_points for index in record_id):
                removal_indices.append(record_id)
        for index in removal_indices:
            del records[index]
        print(records)
    clusters[-1] = list(set(subset).difference(removed_points).difference({0}))
    return clusters

def FOM_features_cosine(event: Event, permutation: Iterable[int]):
    """FOM features for learning the optimal version of the cosine FOM"""
    raise NotImplementedError


def selected_FOM(event: Event, permutation: Iterable[int],
                 start_point:int = 0,
                 start_energy:float = None,
                 Nmi:int = None,
                 eres:float = 1e-3,
                 **kwargs):
    """Features for learning a synthetic FOM"""
    perm = tuple(permutation)
    if len(perm) == 1:
        return None
    if start_energy is None:
        start_energy = np.sum(event.energy_matrix[list(permutation)])
    if Nmi is None:
        Nmi = len(perm)

    # features = {}
    features = np.zeros((11,))

    r_sum_geo = np.abs(event.res_sum_geo(perm,
                                         start_point=start_point,
                                         start_energy=start_energy))

    # features["rsg_mean_1"]  = np.mean(r_sum_geo)
    features[0] = np.mean(r_sum_geo)

    r_cosines = np.abs(event.res_cos(perm,
                                     start_point=start_point,
                                     start_energy=start_energy))

    comp_penalty = np.abs(event.compton_penalty(perm,
                                                start_point=start_point,
                                                start_energy=start_energy))

    # features["c_penalty_sum_1"]  = np.sum(comp_penalty)
    features[1] = np.sum(comp_penalty)

    # features["rc_sum_1_penalty_removed"]   = np.sum(r_cosines*(1 - comp_penalty))
    features[2] = np.sum(r_cosines*(1 - comp_penalty))

    ge_distances = event.ge_distance_perm(perm, start_point=start_point)

    cross_abs = event.linear_attenuation_abs(perm, start_point=start_point,
                                start_energy=start_energy)
    cross_compt = event.linear_attenuation_compt(perm, start_point=start_point,
                                    start_energy=start_energy)
    cross_pair = event.linear_attenuation_pair(perm, start_point=start_point,
                                  start_energy=start_energy)

    cross_total = cross_abs + cross_compt + cross_pair

    # features["-log_p_abs_final"] = -np.log(cross_abs[-1]/cross_total[-1])
    features[3] = -np.log(cross_abs[-1]/cross_total[-1])

    # features["cross_compt_ge_dist_sum"] = np.sum(cross_compt*ge_distances)
    features[4] = np.sum(cross_compt*ge_distances)
    # features["-log_p_compt_max"] = np.max(-np.log(cross_compt/cross_total))
    features[5] = np.max(-np.log(cross_compt/cross_total))

    klein_nishina_rel_sum = event.klein_nishina(perm, start_point=start_point,
                                            start_energy=start_energy,
                                            use_ei=True)

    # features["klein-nishina_rel_sum_sum"] = np.sum(klein_nishina_rel_sum)
    features[6] = np.sum(klein_nishina_rel_sum)

    #%% Now get TANGO features
    e_tango_var = event.estimate_start_energy_sigma_weighted_perm(perm,eres=eres)
    e_sum = sum(event.points[i].e for i in perm)
    start_energy = np.max([e_tango_var, e_sum])

    r_theta_cap = np.abs(event.res_theta_cap(perm,
                                     start_point=start_point,
                                     start_energy=start_energy))

    # features["rth_cap_mean_1_tango"]  = np.mean(r_theta_cap)
    features[7] = np.mean(r_theta_cap)

    cross_compt = event.linear_attenuation_compt(perm, start_point=start_point,
                                    start_energy=start_energy)

    # features["cross_compt_ge_dist_mean_tango"] = np.mean(cross_compt*ge_distances)
    features[8] = np.mean(cross_compt*ge_distances)

    klein_nishina_rel_geo = event.klein_nishina(perm, start_point=start_point,
                                            start_energy=start_energy,
                                            use_ei=False)
    klein_nishina_geo = event.klein_nishina(perm, start_point=start_point,
                                            start_energy=start_energy,
                                            use_ei=False,
                                            relative=False)*RANGE_PROCESS

    # features["-log_klein-nishina_rel_geo_sum_tango"] = np.sum(-np.log(klein_nishina_rel_geo))
    features[9] = np.sum(-np.log(klein_nishina_rel_geo))
    # features["-log_klein-nishina_geo_sum_tango"] = np.sum(-np.log(klein_nishina_geo))
    features[10] = np.sum(-np.log(klein_nishina_geo))

    # weights = np.array([72.23136376610417,
    #                     6.165033062118934,
    #                     16.58736389086943,
    #                     12.565075345443143,
    #                     17.53803792383698,
    #                     11.515862090396933,
    #                     26.732789378541607,
    #                     6.539647682167416,
    #                     9.692628221793408,
    #                     12.596332483268402,
    #                     26.776840589901376])
    weights = np.array([178.12747373389973,
                        14.609762827450487,
                        15.197480477333135,
                        7.626390460173996,
                        5.733775515158737,
                        5.720142738181197,
                        904.9158988958409,
                        3.638575430293514,
                        77.14717485849123,
                        1.7177345247556102,
                        3.5645076928669015])

    return np.dot(features, weights)

#%% Default event and clusters for testing (non-physical)
default_event = Event(0, [Interaction([25, 0, 0], 0.1),
                 Interaction([25, 4, 0], 0.2),
                 Interaction([25, 4, 3], 0.3),
                 Interaction([0, 24, 12], 0.4),
                 Interaction([5, 24, 0], 0.5),
                 Interaction([0, 24, 0], 0.6),
                 Interaction([16, 16, 16], 0.7),
                 Interaction([0, 0, 25], 0.8),
                 Interaction([-16, -16, -16], 0.9),
                ]
            )
default_clusters = {1:[1,2,3], 2:[4,5,6], 3:[7], 4:[8,9]}

#%% Features

def individual_FOM_feature_names():
    """
    Generate an empty FOM feature dictionary for a single cluster
    """
    # from .cluster_tools import cluster_properties
    # feature_names = list(FOM_features(default_event, default_clusters[1]).keys())
    # feature_names.extend([name + '_tango' for name in feature_names])

    # feature_names.append("escape_probability_tango")
    # feature_names.append("-log_escape_probability_tango")
    # feature_names.extend(single_FOM_features(default_event, default_clusters[3]).keys())
    # feature_names.extend(cluster_properties(default_event, default_clusters[1]).features.keys())
    feature_names = list(cluster_FOM_features(default_event, default_clusters[1]).keys())
    feature_dict = {key: 0. for key in feature_names}
    return feature_dict

def multiple_FOM_feature_names():
    """
    Generate an empty FOM feature dictionary for multiple clusters
    """
    feature_names = list(clusters_FOM_features(default_event, default_clusters).keys())
    feature_names.extend(clusters_relative_FOM_features(default_event, default_clusters).keys())
    feature_dict = {key: 0. for key in feature_names}
    return feature_dict

# %% Individual cluster features

def single_FOM_features(event: Event, permutation: Iterable[int],
                        start_point:int = 0,
                        start_energy:float = None,
                        detector:DetectorConfig=default_config) -> Dict:
    """
    Return all of the features for an individual cluster consisting of a single
    interaction

    We can consider these features as entirely separate from the other features
    for clusters, or we can allow them to overlap, or both. This is because the
    values that are computable for a single interaction are just as computable
    for every other kind of cluster. If we separate them, then we are
    acknowledging that singles require wholly different treatment that needs to
    be separately accounted for.

    What features can we actually generate with just the single (no other information available):
    - distance to target
    - linear attenuation coefficient
    - distance to outer shell
    - combinations of distance and attenuation
    - if we had more geometric information, we could get distance to nearest
      exit point of the active detector material
    """
    if isinstance(permutation, int):
        permutation = (permutation,)
    if len(permutation) > 1:
        return {
            "penetration_cm" : 0,
            "edge_cm" : 0,
            "linear_attenuation_cm-1" : 0,
            "energy" : 0,
            "pen_attenuation" : 0,
            "pen_prob_remain" : 0,
            "pen_prob_density" : 0,
            "pen_prob_cumu" : 0,
            "edge_attenuation" : 0,
            "edge_prob_remain" : 0,
            "edge_prob_density" : 0,
            "edge_prob_cumu" : 0,
            "inv_pen" : 0,
            "inv_edge" : 0,
            "interpolated_range" : 0
            }
    linear_attenuation = event.lin_mu_total(permutation=permutation,
                                            start_point=start_point,
                                            start_energy=start_energy)[0]
    distance_to_inside = event.ge_distance[0,permutation[0]]
    distance_to_outside = detector.outer_radius - detector.inner_radius - distance_to_inside
    return {
        "penetration_cm" : distance_to_inside,
        "edge_cm" : distance_to_outside,
        "linear_attenuation_cm-1" : linear_attenuation,
        "energy" : event.energy_matrix[permutation[0]],
        "pen_attenuation" : distance_to_inside * linear_attenuation,
        "pen_prob_remain" : np.exp(-distance_to_inside * linear_attenuation),
        "pen_prob_density" : linear_attenuation * np.exp(-distance_to_inside * linear_attenuation),
        "pen_prob_cumu" : 1 - np.exp(-distance_to_inside * linear_attenuation),
        "edge_attenuation" : distance_to_outside * linear_attenuation,
        "edge_prob_remain" : np.exp(-distance_to_outside * linear_attenuation),
        "edge_prob_density" : linear_attenuation * np.exp(-distance_to_outside * linear_attenuation),
        "edge_prob_cumu" : 1 - np.exp(-distance_to_outside * linear_attenuation),
        "inv_pen" : 1./distance_to_inside,
        "inv_edge" : 1./distance_to_outside,
        "interpolated_range" : singles_depth(event, permutation, detector=detector)
        }

def FOM_features(event: Event, permutation: Iterable[int],
                 start_point:int = 0,
                 start_energy:float = None,
                 Nmi:int = None,
                 eres:float = 1e-3) -> Dict:
    """Features for learning a synthetic FOM"""
    if len(permutation) == 1:
        return None
    if start_energy is None:
        start_energy = np.sum(event.energy_matrix[list(permutation)])
    if Nmi is None:
        Nmi = len(permutation)
    permutation = tuple(permutation)

    features = {}
    # %%
    r_sum_geo = np.abs(event.res_sum_geo(permutation,
                                         start_point=start_point,
                                         start_energy=start_energy))
    r_sum_geo_v = r_sum_geo / np.abs(event.res_sum_geo_sigma(permutation,
                                                             start_point=start_point,
                                                             start_energy=start_energy,
                                                             Nmi=Nmi,
                                                             eres=eres))

    features["rsg_sum_1"]   = np.sum(r_sum_geo)
    features["rsg_sum_1_first"]   = r_sum_geo[0]
    features["rsg_mean_1"]  = np.mean(r_sum_geo)
    features["rsg_mean_1_first"]  = r_sum_geo[0]/len(r_sum_geo)
    features["rsg_wmean_1v"] = np.sum(r_sum_geo_v)/np.sum(r_sum_geo_v/r_sum_geo)
    features["rsg_wmean_1v_first"] = r_sum_geo_v[0]/(r_sum_geo_v[0]/r_sum_geo[0])
    features["rsg_norm_2"]  = np.sqrt(np.sum(r_sum_geo**2))/Nmi

    features["rsg_sum_2"]   = np.sum(r_sum_geo**2)
    features["rsg_sum_2_first"]   = r_sum_geo[0]**2
    features["rsg_mean_2"]  = np.mean(r_sum_geo**2)
    features["rsg_mean_2_first"]  = r_sum_geo[0]**2/len(r_sum_geo)
    features["rsg_wmean_2v"] = np.sum(r_sum_geo_v**2)/np.sum((r_sum_geo_v/r_sum_geo)**2)
    features["rsg_wmean_2v_first"] = r_sum_geo_v[0]**2/((r_sum_geo_v[0]/r_sum_geo[0])**2)

    features["rsg_sum_1v"]  = np.sum(r_sum_geo_v)
    features["rsg_sum_1v_first"] = r_sum_geo_v[0]
    features["rsg_mean_1v"] = np.mean(r_sum_geo_v)
    features["rsg_mean_1v_first"] = r_sum_geo_v[0]/len(r_sum_geo)
    features["rsg_norm_2v"] = np.sqrt(np.sum(r_sum_geo_v**2))/Nmi

    features["rsg_sum_2v"]  = np.sum(r_sum_geo_v**2)
    features["rsg_sum_2v_first"]  = r_sum_geo_v[0]**2
    features["rsg_mean_2v"] = np.mean(r_sum_geo_v**2)
    features["rsg_mean_2v_first"] = r_sum_geo_v[0]**2/len(r_sum_geo)

    # %%
    r_sum_loc = np.abs(event.res_sum_loc(permutation,
                                         start_point=start_point,
                                         start_energy=start_energy))
    r_sum_loc_v = r_sum_loc / np.abs(event.res_sum_loc_sigma(permutation,
                                                             start_point=start_point,
                                                             Nmi=Nmi,
                                                             eres=eres))

    features["rsl_sum_1v"]  = np.sum(r_sum_loc_v)
    features["rsl_mean_1v"] = np.mean(r_sum_loc_v)
    features["rsl_norm_2v"] = np.sqrt(np.mean(r_sum_loc_v**2))/Nmi
    features["rsl_mean_2v"] = np.mean(r_sum_loc_v**2)
    features["rsl_sum_2v"]  = np.sum(r_sum_loc_v**2)

    features["rsl_sum_2"]   = np.sum(r_sum_loc**2)
    features["rsl_mean_2"]  = np.mean(r_sum_loc**2)
    features["rsl_wmean_2v"] = np.sum(r_sum_loc_v**2)/np.sum((r_sum_loc_v/r_sum_loc)**2)

    features["rsl_mean_1"]  = np.mean(r_sum_loc)
    features["rsl_sum_1"]   = np.sum(r_sum_loc)
    features["rsl_norm_2"]  = np.sqrt(np.sum(r_sum_loc**2))/Nmi
    features["rsl_wmean_1v"] = np.sum(r_sum_loc_v)/np.sum(r_sum_loc_v/r_sum_loc)

    # %%
    r_loc_geo = np.abs(event.res_loc_geo(permutation,
                                         start_point=start_point,
                                         start_energy=start_energy))
    r_loc_geo_v = r_loc_geo / np.abs(event.res_loc_geo_sigma(permutation,
                                                             start_point=start_point,
                                                             start_energy=start_energy,
                                                             Nmi=Nmi,
                                                             eres=eres))

    features["rlg_sum_1v"]  = np.sum(r_loc_geo_v)
    features["rlg_mean_1v"] = np.mean(r_loc_geo_v)
    features["rlg_norm_2v"] = np.sqrt(np.mean(r_loc_geo_v**2))/Nmi
    features["rlg_sum_2v"]  = np.sum(r_loc_geo_v**2)
    features["rlg_mean_2v"] = np.mean(r_loc_geo_v**2)

    features["rlg_sum_1"]   = np.sum(r_loc_geo)
    features["rlg_mean_1"]  = np.mean(r_loc_geo)
    features["rlg_norm_2"]  = np.sqrt(np.sum(r_loc_geo**2))/Nmi
    features["rlg_wmean_1v"] = np.sum(r_loc_geo_v)/np.sum(r_loc_geo_v/r_loc_geo)

    features["rlg_sum_2"]   = np.sum(r_loc_geo**2)
    features["rlg_mean_2"]  = np.mean(r_loc_geo**2)
    features["rlg_wmean_2v"] = np.sum(r_loc_geo_v**2)/np.sum((r_loc_geo_v/r_loc_geo)**2)

    # %% Compton penalty
    # 1 - 0 indicator of penalty
    comp_penalty = np.abs(event.compton_penalty(permutation,
                                                start_point=start_point,
                                                start_energy=start_energy))

    # indicator if all of the values require a penalty
    zeroed = (max(comp_penalty.shape) - np.sum(comp_penalty)) < 1

    # continuous penalty value (max(-1 - cos(theta_theo), 0))
    comp_penalty_ell = np.abs(event.compton_penalty_ell1(permutation,
                                                         start_point=start_point,
                                                         start_energy=start_energy))

    features["c_penalty_sum_1"]  = np.sum(comp_penalty)
    features["c_penalty_mean_1"] = np.mean(comp_penalty)
    
    features["c_penalty_ell_sum_1"]  = np.sum(comp_penalty_ell)
    features["c_penalty_ell_mean_1"] = np.mean(comp_penalty_ell)

    features["c_penalty_ell_sum_2"]  = np.sum(comp_penalty_ell**2)
    features["c_penalty_ell_mean_2"] = np.mean(comp_penalty_ell**2)

    # %%
    r_cosines = np.abs(event.res_cos(permutation,
                                     start_point=start_point,
                                     start_energy=start_energy))
    r_cosines_v = r_cosines / np.abs(event.res_cos_sigma(permutation,
                                                         start_point=start_point,
                                                         start_energy=start_energy,
                                                         Nmi=Nmi,
                                                         eres=eres))

    # %%
    r_cosines_cap = np.abs(event.res_cos_cap(permutation,
                                             start_point=start_point,
                                             start_energy=start_energy))
    r_cosines_cap_v = r_cosines_cap / np.abs(event.res_cos_sigma(permutation,
                                                                 start_point=start_point,
                                                                 start_energy=start_energy,
                                                                 Nmi=Nmi,
                                                                 eres=eres))

    # %%
    r_theta = np.abs(event.res_theta(permutation,
                                     start_point=start_point,
                                     start_energy=start_energy))
    r_theta_v = r_theta / np.abs(event.res_theta_sigma(permutation,
                                                       start_point=start_point,
                                                       start_energy=start_energy,
                                                       Nmi=Nmi,
                                                       eres=eres))

    # %%
    r_theta_cap = np.abs(event.res_theta_cap(permutation,
                                     start_point=start_point,
                                     start_energy=start_energy))
    r_theta_cap_v = r_theta_cap / np.abs(event.res_theta_sigma(permutation,
                                                       start_point=start_point,
                                                       start_energy=start_energy,
                                                       Nmi=Nmi,
                                                       eres=eres))

    features["rc_sum_1"]   = np.sum(r_cosines)
    features["rc_mean_1"]  = np.mean(r_cosines)
    features["rc_wmean_1v"] = np.sum(r_cosines_v)/np.sum(r_cosines_v/r_cosines)
    features["rc_norm_2"]  = np.sqrt(np.sum(r_cosines**2))/Nmi

    features["rc_sum_2"]   = np.sum(r_cosines**2)
    features["rc_mean_2"]  = np.mean(r_cosines**2)
    features["rc_wmean_2v"] = np.sum(r_cosines_v**2)/np.sum((r_cosines_v/r_cosines)**2)

    features["rc_sum_1v"]  = np.sum(r_cosines_v)
    features["rc_mean_1v"] = np.mean(r_cosines_v)
    features["rc_norm_2v"] = np.sqrt(np.sum(r_cosines_v**2))/Nmi

    features["rc_sum_2v"]  = np.sum(r_cosines_v**2)
    features["rc_mean_2v"] = np.mean(r_cosines_v**2)

    features["rc_cap_sum_1"]   = np.sum(r_cosines_cap)
    features["rc_cap_mean_1"]  = np.mean(r_cosines_cap)
    features["rc_cap_wmean_1v"] = np.sum(r_cosines_cap_v)/np.sum(r_cosines_cap_v/r_cosines_cap)
    features["rc_cap_norm_2"]  = np.sqrt(np.sum(r_cosines_cap**2))/Nmi

    features["rc_cap_sum_2"]   = np.sum(r_cosines_cap**2)
    features["rc_cap_mean_2"]  = np.mean(r_cosines_cap**2)
    features["rc_cap_wmean_2v"] = np.sum(r_cosines_cap_v**2)/np.sum((r_cosines_cap_v/r_cosines_cap)**2)
    features["rc_sum_1_penalty_removed"]   = np.sum(r_cosines*(1 - comp_penalty))
    features["rc_mean_1_penalty_removed"]  = np.mean(r_cosines*(1 - comp_penalty))
    features["rc_sum_2_penalty_removed"]   = np.sum(r_cosines**2*(1 - comp_penalty))
    features["rc_mean_2_penalty_removed"]  = np.mean(r_cosines**2*(1 - comp_penalty))
    if not zeroed:
        features["rc_wmean_1v_penalty_removed"] = np.sum(
            r_cosines_v*(1 - comp_penalty)
            ) / np.sum(r_cosines_v/r_cosines*(1 - comp_penalty))
        features["rc_wmean_2v_penalty_removed"] = np.sum(
            r_cosines_v**2*(1 - comp_penalty)
            )/np.sum((r_cosines_v/r_cosines*(1 - comp_penalty))**2)
    else:
        features["rc_wmean_1v_penalty_removed"] = 0.
        features["rc_wmean_2v_penalty_removed"] = 0.

    features["rc_cap_sum_1v"]  = np.sum(r_cosines_cap_v)
    features["rc_cap_mean_1v"] = np.mean(r_cosines_cap_v)
    features["rc_cap_norm_2v"] = np.sqrt(np.sum(r_cosines_cap_v**2))/Nmi

    features["rc_cap_sum_2v"]  = np.sum(r_cosines_cap_v**2)
    features["rc_cap_mean_2v"] = np.mean(r_cosines_cap_v**2)

    features["rc_sum_1v_penalty_removed"]  = np.sum(r_cosines_v*(1 - comp_penalty))
    features["rc_mean_1v_penalty_removed"] = np.mean(r_cosines_v*(1 - comp_penalty))
    features["rc_sum_2v_penalty_removed"]  = np.sum(r_cosines_v**2*(1 - comp_penalty))
    features["rc_mean_2v_penalty_removed"] = np.mean(r_cosines_v**2*(1 - comp_penalty))

    features["rth_sum_1"]   = np.sum(r_theta)
    features["rth_mean_1"]  = np.mean(r_theta)
    features["rth_wmean_1v"] = np.sum(r_theta_v)/np.sum(r_theta_v/r_theta)
    features["rth_norm_2"]  = np.sqrt(np.sum(r_theta**2))/Nmi

    features["rth_sum_2"]   = np.sum(r_theta**2)
    features["rth_mean_2"]  = np.mean(r_theta**2)
    features["rth_wmean_2v"] = np.sum(r_theta_v**2)/np.sum((r_theta_v/r_theta)**2)

    features["rth_sum_1v"]  = np.sum(r_theta_v)
    features["rth_mean_1v"] = np.mean(r_theta_v)
    features["rth_norm_2v"] = np.sqrt(np.sum(r_theta_v**2))/Nmi

    features["rth_sum_2v"]  = np.sum(r_theta_v**2)
    features["rth_mean_2v"] = np.mean(r_theta_v**2)

    features["rth_cap_sum_1"]   = np.sum(r_theta_cap)
    features["rth_cap_mean_1"]  = np.mean(r_theta_cap)
    features["rth_cap_wmean_1v"] = np.sum(r_theta_cap_v)/np.sum(r_theta_cap_v/r_theta_cap)
    features["rth_cap_norm_2"]  = np.sqrt(np.sum(r_theta_cap**2))/Nmi

    features["rth_cap_sum_2"]   = np.sum(r_theta_cap**2)
    features["rth_cap_mean_2"]  = np.mean(r_theta_cap**2)
    features["rth_cap_wmean_2v"] = np.sum(r_theta_cap_v**2)/np.sum((r_theta_cap_v/r_theta_cap)**2)

    features["rth_sum_1_penalty_removed"]   = np.sum(r_theta*(1 - comp_penalty))
    features["rth_mean_1_penalty_removed"]  = np.mean(r_theta*(1 - comp_penalty))
    features["rth_sum_2_penalty_removed"]   = np.sum(r_theta**2*(1 - comp_penalty))
    features["rth_mean_2_penalty_removed"]  = np.mean(r_theta**2*(1 - comp_penalty))
    if not zeroed:
        features["rth_wmean_1v_penalty_removed"] = np.sum(
            r_theta_v*(1 - comp_penalty)
            )/np.sum(r_theta_v/r_theta*(1 - comp_penalty))
        features["rth_wmean_2v_penalty_removed"] = np.sum(
            r_theta_v**2*(1 - comp_penalty)
            )/np.sum((r_theta_v/r_theta*(1 - comp_penalty))**2)
    else:
        features["rth_wmean_1v_penalty_removed"] = 0.
        features["rth_wmean_2v_penalty_removed"] = 0.

    features["rth_cap_sum_1v"]  = np.sum(r_theta_cap_v)
    features["rth_cap_mean_1v"] = np.mean(r_theta_cap_v)
    features["rth_cap_norm_2v"] = np.sqrt(np.sum(r_theta_cap_v**2))/Nmi

    features["rth_cap_sum_2v"]  = np.sum(r_theta_cap_v**2)
    features["rth_cap_mean_2v"] = np.mean(r_theta_cap_v**2)

    features["rth_sum_1v_penalty_removed"]  = np.sum(r_theta_v*(1 - comp_penalty))
    features["rth_mean_1v_penalty_removed"] = np.mean(r_theta_v*(1 - comp_penalty))
    features["rth_sum_2v_penalty_removed"]  = np.sum(r_theta_v**2*(1 - comp_penalty))
    features["rth_mean_2v_penalty_removed"] = np.mean(r_theta_v**2*(1 - comp_penalty))

    # %% Distances (Euclidean and Germanium)
    distances = event.distance_perm(permutation, start_point=start_point)
    ge_distances = event.ge_distance_perm(permutation, start_point=start_point)

    features["distances_sum"] = np.sum(distances)
    features["distances_mean"] = np.mean(distances)
    features["ge_distances_sum"] = np.sum(ge_distances)
    features["ge_distances_mean"] = np.mean(ge_distances)

    # %% Attenuation coefficients and cross-sections
    cross_abs = event.linear_attenuation_abs(permutation, start_point=start_point,
                                start_energy=start_energy)
    cross_compt = event.linear_attenuation_compt(permutation, start_point=start_point,
                                    start_energy=start_energy)
    cross_pair = event.linear_attenuation_pair(permutation, start_point=start_point,
                                  start_energy=start_energy)

    cross_total = cross_abs + cross_compt + cross_pair

    features["cross_abs_sum"] = np.sum(cross_abs)
    features["cross_abs_final"] = cross_abs[-1]
    features["cross_abs_mean"] = np.mean(cross_abs)
    features["cross_abs_max"] = np.max(cross_abs)

    features["cross_abs_ge_dist_sum"] = np.sum(cross_abs*ge_distances)
    features["cross_abs_ge_dist_final"] = cross_abs[-1]*ge_distances[-1]
    features["cross_abs_ge_dist_mean"] = np.mean(cross_abs*ge_distances)
    features["cross_abs_ge_dist_max"] = np.max(cross_abs*ge_distances)

    features["cross_abs_dist_sum"] = np.sum(cross_abs*distances)
    features["cross_abs_dist_final"] = cross_abs[-1]*distances[-1]
    features["cross_abs_dist_mean"] = np.mean(cross_abs*distances)
    features["cross_abs_dist_max"] = np.max(cross_abs*distances)

    features["cross_abs_min"] = np.min(cross_abs)
    features["cross_abs_ge_dist_min"] = np.min(cross_abs*ge_distances)
    features["cross_abs_dist_min"] = np.min(cross_abs*distances)

    features["p_abs_sum"] = np.sum(cross_abs/cross_total)
    features["p_abs_final"] = cross_abs[-1]/cross_total[-1]
    features["p_abs_mean"] = np.mean(cross_abs/cross_total)
    features["p_abs_max"] = np.max(cross_abs/cross_total)
    features["-log_p_abs_sum"] = np.sum(-np.log(cross_abs/cross_total))
    features["-log_p_abs_final"] = -np.log(cross_abs[-1]/cross_total[-1])
    features["-log_p_abs_mean"] = np.mean(-np.log(cross_abs/cross_total))
    features["-log_p_abs_max"] = np.max(-np.log(cross_abs/cross_total))

    features["p_abs_min"] = np.min(cross_abs/cross_total)
    features["-log_p_abs_min"] = np.min(-np.log(cross_abs/cross_total))



    features["cross_compt_sum"] = np.sum(cross_compt)
    features["cross_compt_mean"] = np.mean(cross_compt)
    features["cross_compt_max"] = np.max(cross_compt)

    features["cross_compt_ge_dist_sum"] = np.sum(cross_compt*ge_distances)
    features["cross_compt_ge_dist_mean"] = np.mean(cross_compt*ge_distances)
    features["cross_compt_ge_dist_max"] = np.max(cross_compt*ge_distances)

    features["cross_compt_dist_sum"] = np.sum(cross_compt*distances)
    features["cross_compt_dist_mean"] = np.mean(cross_compt*distances)
    features["cross_compt_dist_max"] = np.max(cross_compt*distances)

    features["cross_compt_min"] = np.min(cross_compt)
    features["cross_compt_ge_dist_min"] = np.min(cross_compt*ge_distances)
    features["cross_compt_dist_min"] = np.min(cross_compt*distances)

    features["cross_compt_sum_nonfinal"] = np.sum(cross_compt[:-1])
    features["cross_compt_mean_nonfinal"] = np.mean(cross_compt[:-1])
    features["cross_compt_min_nonfinal"] = np.min(cross_compt[:-1])

    features["cross_compt_dist_sum_nonfinal"] = np.sum(cross_compt[:-1]*distances[:-1])
    features["cross_compt_dist_mean_nonfinal"] = np.mean(cross_compt[:-1]*distances[:-1])
    features["cross_compt_dist_min_nonfinal"] = np.min(cross_compt[:-1]*distances[:-1])

    features["cross_compt_ge_dist_sum_nonfinal"] = np.sum(cross_compt[:-1]*ge_distances[:-1])
    features["cross_compt_ge_dist_mean_nonfinal"] = np.mean(cross_compt[:-1]*ge_distances[:-1])
    features["cross_compt_ge_dist_min_nonfinal"] = np.min(cross_compt[:-1]*ge_distances[:-1])

    features["p_compt_sum"] = np.sum(cross_compt/cross_total)
    features["p_compt_mean"] = np.mean(cross_compt/cross_total)
    features["p_compt_max"] = np.max(cross_compt/cross_total)
    features["p_compt_sum_nonfinal"] = np.sum(cross_compt[:-1]/cross_total[:-1])
    features["p_compt_mean_nonfinal"] = np.mean(cross_compt[:-1]/cross_total[:-1])

    features["-log_p_compt_sum"] = np.sum(-np.log(cross_compt/cross_total))
    features["-log_p_compt_mean"] = np.mean(-np.log(cross_compt/cross_total))
    features["-log_p_compt_max"] = np.max(-np.log(cross_compt/cross_total))
    features["-log_p_compt_sum_nonfinal"] = np.sum(-np.log(cross_compt[:-1]/cross_total[:-1]))
    features["-log_p_compt_mean_nonfinal"] = np.mean(-np.log(cross_compt[:-1]/cross_total[:-1]))

    features["p_compt_min"] = np.min(cross_compt/cross_total)
    features["p_compt_min_nonfinal"] = np.min(cross_compt[:-1]/cross_total[:-1])

    features["-log_p_compt_min"] = np.min(-np.log(cross_compt/cross_total))
    features["-log_p_compt_min_nonfinal"] = np.min(-np.log(cross_compt[:-1]/cross_total[:-1]))

    # # Features for pair production are not as important as other features and
    # # may be misleading to include
    # features["cross_pair_sum"] = np.sum(cross_pair)
    # features["cross_pair_mean"] = np.mean(cross_pair)
    # features["cross_pair_max"] = np.max(cross_pair)
    # features["cross_pair_min"] = np.min(cross_pair)
    # features["cross_pair_dist_sum"] = np.sum(cross_pair*distances)
    # features["cross_pair_dist_mean"] = np.mean(cross_pair*distances)
    # features["cross_pair_dist_max"] = np.max(cross_pair*distances)
    # features["cross_pair_dist_min"] = np.min(cross_pair*distances)
    # features["cross_pair_ge_dist_sum"] = np.sum(cross_pair*ge_distances)
    # features["cross_pair_ge_dist_mean"] = np.mean(cross_pair*ge_distances)
    # features["cross_pair_ge_dist_max"] = np.max(cross_pair*ge_distances)
    # features["cross_pair_ge_dist_min"] = np.min(cross_pair*ge_distances)
    # features["p_pair_sum"] = np.sum(cross_pair/cross_total)
    # features["p_pair_mean"] = np.mean(cross_pair/cross_total)
    # features["p_pair_max"] = np.max(cross_pair/cross_total)
    # features["p_pair_min"] = np.min(cross_pair/cross_total)
    # features["-log_p_pair_sum"] = np.sum(-np.log(cross_pair/cross_total))
    # features["-log_p_pair_mean"] = np.mean(-np.log(cross_pair/cross_total))
    # features["-log_p_pair_max"] = np.max(-np.log(cross_pair/cross_total))
    # features["-log_p_pair_min"] = np.min(-np.log(cross_pair/cross_total))

    features["cross_total_sum"] = np.sum(cross_total)
    features["cross_total_mean"] = np.mean(cross_total)
    features["cross_total_max"] = np.max(cross_total)

    features["cross_total_ge_dist_sum"] = np.sum(cross_total*ge_distances)
    features["cross_total_ge_dist_mean"] = np.mean(cross_total*ge_distances)
    features["cross_total_ge_dist_max"] = np.max(cross_total*ge_distances)

    features["cross_total_dist_sum"] = np.sum(cross_total*distances)
    features["cross_total_dist_mean"] = np.mean(cross_total*distances)
    features["cross_total_dist_max"] = np.max(cross_total*distances)

    features["cross_total_min"] = np.min(cross_total)
    features["cross_total_ge_dist_min"] = np.min(cross_total*ge_distances)
    features["cross_total_dist_min"] = np.min(cross_total*distances)

    # %% Klein Nishina features
    klein_nishina_rel_sum = event.klein_nishina(permutation, start_point=start_point,
                                            start_energy=start_energy,
                                            use_ei=True)
    klein_nishina_rel_geo = event.klein_nishina(permutation, start_point=start_point,
                                            start_energy=start_energy,
                                            use_ei=False)
    klein_nishina_sum = event.klein_nishina(permutation, start_point=start_point,
                                            start_energy=start_energy,
                                            use_ei=True,
                                            relative=False)*RANGE_PROCESS
    klein_nishina_geo = event.klein_nishina(permutation, start_point=start_point,
                                            start_energy=start_energy,
                                            use_ei=False,
                                            relative=False)*RANGE_PROCESS

    features["klein-nishina_rel_sum_sum"] = np.sum(klein_nishina_rel_sum)
    features["klein-nishina_rel_sum_mean"] = np.mean(klein_nishina_rel_sum)
    features["klein-nishina_rel_sum_max"] = np.max(klein_nishina_rel_sum)
    features["klein-nishina_rel_sum_min"] = np.min(klein_nishina_rel_sum)

    features["klein-nishina_sum_sum"] = np.sum(klein_nishina_sum)
    features["klein-nishina_sum_mean"] = np.mean(klein_nishina_sum)
    features["klein-nishina_sum_max"] = np.max(klein_nishina_sum)
    features["klein-nishina_sum_min"] = np.min(klein_nishina_sum)

    features["-log_klein-nishina_rel_sum_sum"] = np.sum(-np.log(klein_nishina_rel_sum))
    features["-log_klein-nishina_rel_sum_mean"] = np.mean(-np.log(klein_nishina_rel_sum))
    features["-log_klein-nishina_rel_sum_max"] = np.max(-np.log(klein_nishina_rel_sum))
    features["-log_klein-nishina_rel_sum_min"] = np.min(-np.log(klein_nishina_rel_sum))

    features["-log_klein-nishina_sum_sum"] = np.sum(-np.log(klein_nishina_sum))
    features["-log_klein-nishina_sum_mean"] = np.mean(-np.log(klein_nishina_sum))
    features["-log_klein-nishina_sum_max"] = np.max(-np.log(klein_nishina_sum))
    features["-log_klein-nishina_sum_min"] = np.min(-np.log(klein_nishina_sum))

    features["klein-nishina_rel_geo_sum"] = np.sum(klein_nishina_rel_geo)
    features["klein-nishina_rel_geo_mean"] = np.mean(klein_nishina_rel_geo)
    features["klein-nishina_rel_geo_max"] = np.max(klein_nishina_rel_geo)
    features["klein-nishina_rel_geo_min"] = np.min(klein_nishina_rel_geo)

    features["klein-nishina_geo_sum"] = np.sum(klein_nishina_geo)
    features["klein-nishina_geo_mean"] = np.mean(klein_nishina_geo)
    features["klein-nishina_geo_max"] = np.max(klein_nishina_geo)
    features["klein-nishina_geo_min"] = np.min(klein_nishina_geo)

    features["-log_klein-nishina_rel_geo_sum"] = np.sum(-np.log(klein_nishina_rel_geo))
    features["-log_klein-nishina_rel_geo_mean"] = np.mean(-np.log(klein_nishina_rel_geo))
    features["-log_klein-nishina_rel_geo_max"] = np.max(-np.log(klein_nishina_rel_geo))
    features["-log_klein-nishina_rel_geo_min"] = np.min(-np.log(klein_nishina_rel_geo))

    features["-log_klein-nishina_geo_sum"] = np.sum(-np.log(klein_nishina_geo))
    features["-log_klein-nishina_geo_mean"] = np.mean(-np.log(klein_nishina_geo))
    features["-log_klein-nishina_geo_max"] = np.max(-np.log(klein_nishina_geo))
    features["-log_klein-nishina_geo_min"] = np.min(-np.log(klein_nishina_geo))

    return features

#%% Sequence features
def FOM_sequence_features(
    event: Event,
    permutation: Iterable[int],
    start_point:int = 0,
    start_energy:float = None,
    Nmi:int = None,
    eres:float = 1e-3
    ) -> Dict:
    """Features for learning a synthetic FOM using data sequences"""
    if len(permutation) == 1:
        return None
    if start_energy is None:
        start_energy = np.sum(event.energy_matrix[list(permutation)])
    if Nmi is None:
        Nmi = len(permutation)
    permutation = tuple(permutation)

    features = {}

    r_sum_geo = np.abs(event.res_sum_geo(permutation,
                                         start_point=start_point,
                                         start_energy=start_energy))
    r_sum_geo_std = np.abs(event.res_sum_geo_sigma(permutation,
                                                        start_point=start_point,
                                                        start_energy=start_energy,
                                                        Nmi=Nmi,
                                                        eres=eres))
    r_sum_geo_v = r_sum_geo / r_sum_geo_std

    features["rsg_1"] = r_sum_geo
    features["rsg_mean_1"] = r_sum_geo/len(r_sum_geo)
    features["rsg_1v"] = r_sum_geo_v

    features["rsg_2"] = r_sum_geo**2
    features["rsg_mean_2"] = r_sum_geo**2/len(r_sum_geo)
    features["rsg_2v"] = r_sum_geo_v**2

    features["rsg_std"] = r_sum_geo_std


    r_sum_loc = np.abs(event.res_sum_loc(permutation,
                                         start_point=start_point,
                                         start_energy=start_energy))
    r_sum_loc_std = np.abs(event.res_sum_loc_sigma(permutation,
                                                   start_point=start_point,
                                                   Nmi=Nmi,
                                                   eres=eres))
    r_sum_loc_v = r_sum_loc / r_sum_loc_std

    features["rsl_1"] = r_sum_loc
    features["rsl_mean_1"] = r_sum_loc/len(r_sum_loc)
    features["rsl_1v"] = r_sum_loc_v

    features["rsl_2"] = r_sum_loc**2
    features["rsl_mean_2"] = r_sum_loc**2/len(r_sum_loc)
    features["rsl_2v"] = r_sum_loc_v**2

    features["rsl_std"] = r_sum_loc_std


    r_loc_geo = np.abs(event.res_loc_geo(permutation,
                                         start_point=start_point,
                                         start_energy=start_energy))
    r_loc_geo_std = np.abs(event.res_loc_geo_sigma(permutation,
                                                   start_point=start_point,
                                                   start_energy=start_energy,
                                                   Nmi=Nmi,
                                                   eres=eres))
    r_loc_geo_v = r_loc_geo / r_loc_geo_std

    features["rlg_1"] = r_loc_geo
    features["rlg_mean_1"] = r_loc_geo/len(r_loc_geo)
    features["rlg_1v"] = r_loc_geo_v

    features["rlg_2"] = r_loc_geo**2
    features["rlg_mean_2"] = r_loc_geo**2/len(r_loc_geo)
    features["rlg_2v"] = r_loc_geo_v**2

    features["rlg_std"] = r_loc_geo_std


    r_cosines = np.abs(event.res_cos(permutation,
                                     start_point=start_point,
                                     start_energy=start_energy))
    r_cosines_std = np.abs(event.res_cos_sigma(permutation,
                                               start_point=start_point,
                                               start_energy=start_energy,
                                               Nmi=Nmi,
                                               eres=eres))
    r_cosines_v = r_cosines / r_cosines_std

    features["rc_1"] = r_cosines
    features["rc_mean_1"] = r_cosines/len(r_cosines)
    features["rc_1v"] = r_cosines_v

    features["rc_2"] = r_cosines**2
    features["rc_mean_2"] = r_cosines**2/len(r_cosines)
    features["rc_2v"] = r_cosines_v**2

    features["rc_std"] = r_cosines_std


    r_theta = np.abs(event.res_theta(permutation,
                                     start_point=start_point,
                                     start_energy=start_energy))
    r_theta_std = np.abs(event.res_theta_sigma(permutation,
                                               start_point=start_point,
                                               start_energy=start_energy,
                                               Nmi=Nmi,
                                               eres=eres))
    r_theta_v = r_theta / r_theta_std

    features["rth_1"] = r_theta
    features["rth_mean_1"] = r_theta/len(r_theta)
    features["rth_1v"] = r_theta_v

    features["rth_2"] = r_theta**2
    features["rth_mean_2"] = r_theta**2/len(r_theta)
    features["rth_2v"] = r_theta_v**2

    features["rth_std"] = r_theta_std


    comp_penalty = np.abs(event.compton_penalty(permutation,
                                                start_point=start_point,
                                                start_energy=start_energy))

    zeroed = (max(comp_penalty.shape) - np.sum(comp_penalty)) < 1

    features["c_penalty_1"]  = comp_penalty
    features["c_penalty_mean_1"] = comp_penalty/len(comp_penalty)

    comp_penalty_ell = np.abs(event.compton_penalty_ell1(permutation,
                                                         start_point=start_point,
                                                         start_energy=start_energy))

    features["c_penalty_ell_1"]  = comp_penalty_ell
    features["c_penalty_ell_mean_1"] = comp_penalty_ell/len(comp_penalty_ell)


    r_cosines_cap = np.abs(event.res_cos_cap(permutation,
                                             start_point=start_point,
                                             start_energy=start_energy))
    r_cosines_cap_v = r_cosines_cap / r_cosines_std

    features["rc_cap_1"] = r_cosines_cap
    features["rc_cap_mean_1"] = r_cosines_cap/len(r_cosines_cap)
    features["rc_cap_1v"] = r_cosines_cap_v

    features["rc_cap_2"] = r_cosines_cap**2
    features["rc_cap_mean_2"] = r_cosines_cap**2/len(r_cosines_cap)
    features["rc_cap_2v"] = r_cosines_cap_v**2


    r_theta_cap = np.abs(event.res_theta_cap(permutation,
                                     start_point=start_point,
                                     start_energy=start_energy))
    r_theta_cap_v = r_theta_cap / r_theta_std

    features["rth_cap_1"] = r_theta_cap
    features["rth_cap_mean_1"] = r_theta_cap/len(r_theta_cap)
    features["rth_cap_1v"] = r_theta_cap_v

    features["rth_cap_2"] = r_theta_cap**2
    features["rth_cap_mean_2"] = r_theta_cap**2/len(r_theta_cap)
    features["rth_cap_2v"] = r_theta_cap_v**2


    distances = event.distance_perm(permutation, start_point=start_point)
    ge_distances = event.ge_distance_perm(permutation, start_point=start_point)

    features["distances"] = distances
    features["distances_mean"] = distances/len(distances)

    features["ge_distances"] = ge_distances
    features["ge_distances_mean"] = ge_distances/len(ge_distances)


    cross_abs = event.linear_attenuation_abs(permutation, start_point=start_point,
                                start_energy=start_energy)
    cross_compt = event.linear_attenuation_compt(permutation, start_point=start_point,
                                    start_energy=start_energy)
    cross_pair = event.linear_attenuation_pair(permutation, start_point=start_point,
                                  start_energy=start_energy)

    cross_total = cross_abs + cross_compt + cross_pair

    features["cross_abs"] = cross_abs
    features["cross_abs_mean"] = cross_abs/len(cross_abs)

    features["cross_abs_ge_dist"] = cross_abs*ge_distances
    features["cross_abs_ge_dist_mean"] = cross_abs*ge_distances/len(cross_abs)

    features["cross_abs_dist"] = cross_abs*distances
    features["cross_abs_dist_mean"] = cross_abs*distances/len(cross_abs)

    features["p_abs"] = cross_abs/cross_total
    features["p_abs_mean"] = cross_abs/cross_total/len(cross_abs)

    features["-log_p_abs"] = -np.log(cross_abs/cross_total)
    features["-log_p_abs_mean"] = -np.log(cross_abs/cross_total)/len(cross_abs)


    features["cross_compt"] = cross_compt
    features["cross_compt_mean"] = cross_compt/len(cross_compt)

    features["cross_compt_ge_dist"] = cross_compt*ge_distances
    features["cross_compt_ge_dist_mean"] = cross_compt*ge_distances/len(cross_compt)

    features["cross_compt_dist"] = cross_compt*distances
    features["cross_compt_dist_mean"] = cross_compt*distances/len(cross_compt)

    features["p_compt"] = cross_compt/cross_total
    features["p_compt_mean"] = cross_compt/cross_total/len(cross_compt)

    features["-log_p_compt"] = -np.log(cross_compt/cross_total)
    features["-log_p_compt_mean"] = -np.log(cross_compt/cross_total)/len(cross_compt)


    # # Features for pair production are not as important as other features and
    # # may be misleading to include
    # features["cross_pair"] = cross_pair
    # features["cross_pair_mean"] = cross_pair/len(cross_pair)

    # features["cross_pair_ge_dist"] = cross_pair*ge_distances
    # features["cross_pair_ge_dist_mean"] = cross_pair*ge_distances/len(cross_pair)

    # features["cross_pair_dist"] = cross_pair*distances
    # features["cross_pair_dist_mean"] = cross_pair*distances/len(cross_pair)

    # features["p_pair"] = cross_pair/cross_total
    # features["p_pair_mean"] = cross_pair/cross_total/len(cross_pair)

    # features["-log_p_pair"] = -np.log(cross_pair/cross_total)
    # features["-log_p_pair_mean"] = -np.log(cross_pair/cross_total)/len(cross_pair)

    features["cross_total"] = cross_total
    features["cross_total_mean"] = cross_total/len(cross_total)

    features["cross_total_ge_dist"] = cross_total*ge_distances
    features["cross_total_ge_dist_mean"] = cross_total*ge_distances/len(cross_total)

    features["cross_total_dist"] = cross_total*distances
    features["cross_total_dist_mean"] = cross_total*distances/len(cross_total)

    klein_nishina_rel_sum = event.klein_nishina(permutation, start_point=start_point,
                                            start_energy=start_energy,
                                            use_ei=True)
    klein_nishina_rel_geo = event.klein_nishina(permutation, start_point=start_point,
                                            start_energy=start_energy,
                                            use_ei=False)
    klein_nishina_sum = event.klein_nishina(permutation, start_point=start_point,
                                            start_energy=start_energy,
                                            use_ei=True,
                                            relative=False)*RANGE_PROCESS
    klein_nishina_geo = event.klein_nishina(permutation, start_point=start_point,
                                            start_energy=start_energy,
                                            use_ei=False,
                                            relative=False)*RANGE_PROCESS

    features["klein-nishina_rel_sum"] = klein_nishina_rel_sum
    features["klein-nishina_rel_sum_mean"] = klein_nishina_rel_sum/len(klein_nishina_rel_sum)

    features["-log_klein-nishina_rel_sum"] = -np.log(klein_nishina_rel_sum)
    features["-log_klein-nishina_rel_sum_mean"] = -np.log(klein_nishina_rel_sum)/len(klein_nishina_rel_sum)


    features["klein-nishina_rel_geo"] = klein_nishina_rel_geo
    features["klein-nishina_rel_geo_mean"] = klein_nishina_rel_geo/len(klein_nishina_rel_geo)

    features["-log_klein-nishina_rel_geo"] = -np.log(klein_nishina_rel_geo)
    features["-log_klein-nishina_rel_geo_mean"] = -np.log(klein_nishina_rel_geo)/len(klein_nishina_rel_geo)


    features["klein-nishina_sum"] = klein_nishina_sum
    features["klein-nishina_sum_mean"] = klein_nishina_sum/len(klein_nishina_sum)

    features["-log_klein-nishina_sum"] = -np.log(klein_nishina_sum)
    features["-log_klein-nishina_sum_mean"] = -np.log(klein_nishina_sum)/len(klein_nishina_sum)


    features["klein-nishina_geo"] = klein_nishina_geo
    features["klein-nishina_geo_mean"] = klein_nishina_geo/len(klein_nishina_geo)

    features["-log_klein-nishina_geo"] = -np.log(klein_nishina_geo)
    features["-log_klein-nishina_geo_mean"] = -np.log(klein_nishina_geo)/len(klein_nishina_geo)

    return features

def escape_probability(p_imo:Interaction, p_i:Interaction,
                       E_x:float, detector:DetectorConfig=default_config):
    """
    point location of last point before scattering out
    direction of scatter axis (previous point to current point)
    opening angle (CSF)
    Ex excess energy for cross section values
    """
    point = p_i.x
    direction = p_i.x-p_imo.x
    direction /= np.linalg.norm(direction)
    opening_angle = theta_theor(E_x + p_i.e, E_x)
    theor_cos = cos_theor(E_x + p_i.e, E_x)
    if theor_cos < -1:
        # return (0., 0)
        return 0.

    c = cone(point, direction, opening_angle, detector.outer_radius)
    linear_attenuation = lin_att_total(np.array([E_x]))[0]
    def probability(theta):
        # TODO - not clear if its faster to exploit symmetry and only integrate
        # over pi instead of 2*pi
        return np.exp(- linear_attenuation * c.ray_lengths(theta))/2/np.pi
    return integrate.quad(probability, 0, 2*np.pi)[0]

def escape_prob_cluster(event:Event, perm:tuple, start_energy:float,
                start_point:int=0,
                detector:DetectorConfig=default_config):
    """
    Get the escape probability of a cluster by looking at the final interactions
    """
    full_perm = tuple([start_point] + list(perm))
    E_x = start_energy - event.energy_sum(perm)
    if E_x <= 0:
        return 0.
    return escape_probability(event.points[full_perm[-2]], event.points[full_perm[-1]],
                             E_x, detector=detector)

#TODO - emittance likelihood for a ray: basically backtracking where we look at
#a cone for the ray in the reverse direction and determine if the ray is likely
#to have come from the origin? I'm not sure how best to do this, but should be a
#nice feature for evaluating a gamma-ray track. The backtracking angle is the
#same as the forward theta_theoretical...

def cluster_FOM_features(event: Event, permutation: Iterable[int],
                         start_point:int = 0,
                         start_energy:float = None,
                         Nmi:int = None,
                         eres:float = 1e-3,
                         detector:DetectorConfig=default_config) -> Dict:
    """Return all of the features for an individual cluster

    Logic:
    Return FOM_features for cluster
    Return FOM_features for cluster with TANGO energy
    If a single: return the single features

    Want the same output structure regardless if it is single or a full cluster
    """
    from .cluster_tools import cluster_properties

    features = {}

    if start_energy is None:
        start_energy = np.sum(event.energy_matrix[list(permutation)])

    props = cluster_properties(event, permutation)
    property_features = props.features

    features = features | property_features

    singles_features =  single_FOM_features(event, permutation, start_point=start_point,
                                   start_energy=start_energy,
                                   detector=detector)

    features = singles_features | features

    if len(permutation) == 1:
        return features

    energy_sum_features = FOM_features(event, permutation,
                                       start_point=start_point,
                                       start_energy=start_energy,
                                       Nmi=Nmi,
                                       eres=eres)
    tango_energy = max(event.estimate_start_energy_sigma_weighted_perm(permutation,
                                                                       start_point=start_point,
                                                                       eres=eres),
                       start_energy)

    if tango_energy > start_energy:
        tango_energy_features = FOM_features(event, permutation,
                                            start_point=start_point,
                                            start_energy=tango_energy,
                                            Nmi=Nmi,
                                            eres=eres)
    else:
        tango_energy_features = deepcopy(energy_sum_features)

    new_tango_features = {key + "_tango": value for key, value in tango_energy_features.items()}

    new_tango_features["escape_probability_tango"] = escape_prob_cluster(event,
                                                                      permutation,
                                                                      tango_energy,
                                                                      detector=detector)
    new_tango_features["-log_escape_probability_tango"] = -np.log(new_tango_features["escape_probability_tango"])
    features = energy_sum_features | new_tango_features | features
    # features = {**default_features, **features}
    return features


# %% Multiple clusters features
def clusters_relative_FOM_features(event: Event, clusters: Dict[int,Iterable[int]]) -> Dict:
    """
    When we have multiple clusters, we can have relative features for each
    cluster (we don't need to consider them totally in isolation as we would do
    otherwise). In the case that we are only looking at a single cluster, the
    relative features will not charge (only if we look at the shape of the
    cluster will it change since different orders have different shapes).
    """
    from .cluster_tools import cluster_pdist
    cluster_distances_ge = cluster_pdist(event, clusters, method = 'single', metric= 'germanium')
    cluster_distances_ge[cluster_distances_ge == 0] = np.inf
    cluster_distances_euc = cluster_pdist(event, clusters, method = 'single', metric= 'euclidean')
    cluster_distances_euc[cluster_distances_euc == 0] = np.inf
    isolation_ge = 1/np.min(cluster_distances_ge, axis = 1)
    isolation_euc = 1/np.min(cluster_distances_euc, axis = 1)

    features = {"isolation_ge_single": 0.,
                "isolation_ge": 0.,
                "isolation_euc_single": 0.,
                "isolation_euc": 0.}
    for i, cluster in enumerate(clusters.values()):
        if len(cluster) == 1:
            # we have a single, add the isolation to the singles feature
            features["isolation_ge_single"] += isolation_ge[i]
            features["isolation_euc_single"] += isolation_euc[i]
        else:
            # we don't have a single, add the isolation to the other feature
            features["isolation_ge"] += isolation_ge[i]
            features["isolation_euc"] += isolation_euc[i]
    return features

def clusters_FOM_features(event: Event, clusters: Dict[int,Iterable[int]]) -> Dict:
    """
    We want to compute features for any set of clusters that we can possibly
    gather together from an event (i.e., we want to get features for each
    cluster and combine them).

    Logic:
    - Compute features for each cluster individual
    - Add them up as a vector
    - Add in additional relative features (features depending on the distances
    and orientations of clusters with respect to one another)
    - Split the sum and average because the number of clusters may change (changing the average)
    """
    features_list = [cluster_FOM_features(event, cluster) for cluster in clusters.values()]
    num_interactions = sum((len(cluster) for cluster in clusters.values()))
    features = features_list[0]
    for f in features_list[1:]:
        for key in f.keys():
            features[key] = features.get(key, 0.0) + f[key]
    avg_features = {}
    for key in features.keys():
        avg_features[key + '_avg'] = features[key]/len(clusters)
    interaction_weighted_features = {}
    for key in features.keys():
        interaction_weighted_features[key + "_int_weighted"] = 0
    for cluster, f in zip(clusters.values(), features_list):
        for key in f.keys():
            interaction_weighted_features[key + "_int_weighted"] += \
                len(cluster)*f[key]/num_interactions
    relative_features = clusters_relative_FOM_features(event, clusters)
    features = features | avg_features | interaction_weighted_features | relative_features
    return features

default_ind_feature_names = individual_FOM_feature_names()
default_mul_feature_names = multiple_FOM_feature_names()

def get_all_features_cluster(event: Event,
                             cluster: Iterable[int],
                             start_point: int = 0,
                             start_energy: float = None) -> Dict:
    """
    Get all of the possible features for an individual cluster. Relative
    features are not included
    """
    return {**default_ind_feature_names, **cluster_FOM_features(event=event,
                                                                permutation=cluster,
                                                                start_point=start_point,
                                                                start_energy=start_energy)}

def get_all_features_clusters(event: Event,
                              clusters: Dict[int, Iterable[int]]) -> Dict:
    """
    Get all of the possible features for a group of clusters. Relative features
    are included and the values from individual clusters are summed by feature
    """
    return {**default_mul_feature_names, **clusters_FOM_features(event=event,
                                                                 clusters=clusters)}


# %% Tensor FOM and tensor reduction FOM

def tensor_FOM(
    event: Event,
    permutation: Tuple,
    start_point: int = 0,
    agg_method: Callable = np.sum
    ) -> float:
    """
    Given a tensor of transition quality estimates, compute the aggregation of
    the transitions in the proposed permutation.
    """
    full_perm = [start_point] + list(permutation)
    return agg_method(event.quality_tensor[full_perm[:-2], full_perm[1:-1], full_perm[2:]])

def reduction_FOM(
    event: Event,
    permutation: Tuple,
    start_point: int = 0,
    agg_method: Callable = np.sum
    ) -> float:
    """
    Given a tensor reduction of transition quality estimates, compute the
    aggregation of the transitions in the proposed permutation.
    """
    full_perm = [start_point] + list(permutation)
    return agg_method(event.reduction[full_perm[:-1], full_perm[1:]])
