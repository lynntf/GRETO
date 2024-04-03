"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Forward estimate of tracking using geometry to pre-screen interactions
"""
from typing import List

from greto.event_class import Event
from greto.physics import theta_theor


def interactions_in_cone(
    event: Event,
    pt_idx: int,
    energy: float,
    angle_tol: float,
    max_distance: float,
    start_pt: int = 0,
    energy_tol: float = 0.01,
) -> List[int]:
    """
    Given an Event, an interaction point in the event, and an energy level,
    compute the points which make approximately the expected angle given
    by the compton scattering formula.
    The gamma ray is assumed to emanate from the origin unless the start_pt
    argument is set to an index of a different point which will be the starting
    point then.

    Args:
        event (Event): The gamma ray Event
        pt_idx (int): The index of the point that the gamma ray is hitting
        energy (float): The energy the gamma ray is starting with
        angle_tol (float): The allowable error tolerance between expected
            and actual angle
        max_distance (float): The maximum distance the gamma ray is allowed
            to travel through the germanium
        start_pt (int): If None, this means the gamma ray emanates from the
            origin, if this is an integer, this is the index of the point
            the gamma ray emanates from.
        energy_tol (float): If the amount of energy deposited at the pt_idx
            point is about equal to energy, then it is assumed the the gamma
            ray is absorbed at the interaction. If the energy deposited is
            within energy_tol of the passed in energy, then no interactions
            are returned, as this is an absorption.

    TODO - Modify for backward tracking (should be fairly straightforward)
    """

    pt = event.points[pt_idx]
    if abs(energy - pt.e) < energy_tol:
        return []

    # Compton Scattering Formula
    pred_angle = theta_theor(energy, energy - pt.e)
    good_interactions = []
    for i in range(1, len(event.points)):
        if i == pt_idx:
            continue

        # Skip any interactions which are farther than the max distance
        if event.ge_distance[pt_idx, i] > max_distance:
            continue

        act_angle = event.theta_act_perm([pt_idx, i], start_point=start_pt)

        # If the angles are close enough, we accept the interaction
        if abs(pred_angle - act_angle) <= angle_tol:
            good_interactions.append(i)
    return good_interactions


def construct_trajectories_(
    event: Event,
    pt_idx: int,
    energy: float,
    angle_tol: float,
    max_distance: float,
    energy_tol: float,
    curr_path: List[int],
):
    """
    Function that recursively searches through possible cone defined
    trajectories

    TODO - Looks like there are problems with the return format (input is just a
    list, but output appears to be a list of lists? Not sure if that is correct)
    """
    pt = event.points[pt_idx]

    # Not enough energy to continue
    if abs(energy - pt.e) < energy_tol:
        return [curr_path]

    curr_trajectories = []

    start_pt = event.points[curr_path[-2]]
    next_pt_candidates = interactions_in_cone(
        event, pt_idx, energy, angle_tol, max_distance, start_pt
    )
    for candidate in next_pt_candidates:
        if candidate in curr_path:
            continue
        candidate_pt = event.points[candidate]
        distance = event.ge_distance[candidate, pt_idx]

        # Too much energy at candidate point, or too far
        if pt.e + candidate_pt.e > energy + energy_tol or distance > max_distance:
            continue

        for trajectory in construct_trajectories_(
            event,
            candidate,
            energy - pt.e,
            angle_tol,
            max_distance - distance,
            energy_tol,
            curr_path + [candidate],
        ):
            curr_trajectories.append(trajectory)
    return curr_trajectories


def construct_trajectories(
    event: Event,
    pt_idx: int,
    energy: float,
    angle_tol: float,
    max_distance: float,
    energy_tol: float,
):
    """
    Given an Event, an interaction point in the event, and an energy level,
    compute all trajectories which start at origin, hit the given point first,
    and then hit some collection of other points while satisfying that the
    compton scattering angle and actual angles are less than angle_tol, the
    total distance traveled is less than max_distance, and the total energy
    deposited is within energy_tol of energy.

    This is done by following the trajectory of the gamma ray and searching
    all paths which satisfy the angle tolerance requirements.

    Args:
        event (Event): The gamma ray Event
        pt_idx (int): The index of the point that the gamma ray is hitting
        energy (float): The energy the gamma ray is starting with
        angle_tol (float): The allowable error tolerance between expected
            and actual angle
        max_distance (float): The maximum distance the gamma ray is allowed
            to travel through the germanium
        start_pt (int): If None, this means the gamma ray emanates from the
            origin, if this is an integer, this is the index of the point
            the gamma ray emanates from.
        energy_tol (float): If the amount of energy deposited at the pt_idx
            point is about equal to energy, then it is assumed the the gamma
            ray is absorbed at the interaction. If the energy deposited is
            within energy_tol of the passed in energy, then no interactions
            are returned, as this is an absorption.
    """
    pt = event.points[pt_idx]
    if pt.e > energy + energy_tol:
        return []

    return construct_trajectories_(
        event, pt_idx, energy, angle_tol, max_distance, energy_tol, [0, pt_idx]
    )
