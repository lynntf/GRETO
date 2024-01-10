"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Event processing functions
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.transform import Rotation

from .event_class import Event
from .geometry import ge_distance, germanium_extension
from .interaction_class import Interaction
from .utils import perm_to_transition


def merge_events(list_of_events:List[Event],
                 list_of_clusters:List[Dict[int,List[int]]]= None) -> Tuple[Event, Dict]:
    """
    Merge a collection of Events into a single event

    Arg:
        list_of_events: the events to be merged
        list_of_clusters: the clusters to be merged
    """
    merged_id = (event.id for event in list_of_events)
    merged_points = []
    if list_of_clusters is None:
        for event in list_of_events:
            merged_points.extend(event.hit_points)
        return Event(merged_id, merged_points)
    new_clusters = {}
    new_index = 1
    offset = 0
    for event, clusters in zip(list_of_events, list_of_clusters):
        merged_points.extend(event.hit_points)
        for cluster in clusters.values():
            new_clusters[new_index] = [index + offset for index in cluster]
            new_index += 1
        offset = len(merged_points)
    return (Event(merged_id, merged_points), new_clusters)

def split_event(event: Event, clustering: Dict) -> List[Event]:
    """
    Split the Event into sub-events based on the given clustering of points.

    Args:
        clustering : A dictionary mapping cluster ids to lists of
            point indexes
    """
    events = []
    for c, cluster in clustering.items():
        if isinstance(event.id, tuple):
            new_event_id = event.id
        else:
            new_event_id = (int(event.id), c)
        events.append(Event(
            # str(event.id) + '_' + str(c),
            new_event_id,
            [event.points[i] for i in cluster]),
        )
    return events

def split_event_clusters(event: Event, clustering: Dict) -> Tuple[List[Event], List[Dict]]:
    """
    Split the Event into sub-events based on the given clustering of points.

    Args:
        clustering : A dictionary mapping cluster ids to lists of
            point indexes
    """
    events = []
    clusters = []
    for c, cluster in clustering.items():
        if isinstance(event.id, tuple):
            new_event_id = event.id
        else:
            new_event_id = (int(event.id), c)
        events.append(Event(
            # str(event.id) + '_' + str(c),
            new_event_id,
            [event.points[i] for i in sorted(cluster)]),
        )
        clusters.append({c: list(np.argsort(cluster) + 1)})
    return events, clusters

def subset(event: Event, indices: Iterable, event_id: Any=None,
           reset_indices: bool = False) -> Event:
    """
    Return an Event using a subset of the events given by the indexes.

    Args:
        indices: The indexes of the Interactions to be in the subset
        reset_idxs (bool): If True, reset the index associated with each
            Interaction
    """
    if event_id is None:
        event_id = event.id
    sub_event = Event(event_id, [deepcopy(event.points[i]) for i in indices])
    if reset_indices:
        for i, point in enumerate(sub_event.points):
            point.id = i
    return sub_event

def remove_interactions(event: Event, removal_indices: Iterable, clusters: Dict = None):
    """
    Remove interactions from an event by index and correct cluster indices
    """
    shift = 0
    index_map = {}
    new_points = []
    for i, point in enumerate(event.points):
        if i in removal_indices:
            shift += 1
        else:
            new_points.append(point)
        index_map[i] = i - shift
    if clusters is not None:
        new_clusters = {}
        for cluster_idx in clusters:
            new_clusters[cluster_idx] = []
            for i, index in enumerate(clusters[cluster_idx]):
                if index in removal_indices:
                    continue
                new_clusters[cluster_idx].append(index_map[index])
            if len(new_clusters[cluster_idx]) == 0:
                del new_clusters[cluster_idx]
        return Event(event.id, new_points), new_clusters
    return Event(event.id, new_points)

#%% Data augmentation (maintain distance and angle)

def flatten_cluster(event:Event, permutation:list[int],
                   start_point:int=0,
                   randomize: bool = False,
                   randomize_initial_heading:bool = False,
                   randomize_heading:bool = False,
                   rng:np.random.RandomState = None):
    """
    Transform from 3D cluster to 2D, maintaining angles, for creating images/figures
    
    Each scatter can be one of two directions:
        pick the one that maintains the radial distance the best?
    """
    if randomize:
        randomize_heading = True
        randomize_initial_heading = True
        if rng is None:
            rng = np.random.RandomState()
    angles = np.arccos(event.cos_act_perm(permutation, start_point=start_point))
    ge_distances = event.ge_distance_perm(permutation, start_point=start_point)
    init_direction = event.points[permutation[0]].x - event.points[start_point].x
    r = np.linalg.norm(init_direction)
    if randomize_initial_heading:
        init_theta = rng.uniform(0.0, 2*np.pi)
    else:
        init_theta = np.arctan2(init_direction[2], init_direction[1])
    theta = np.copy(init_theta)
    point_x = []
    point_x.append(np.array([r*np.cos(theta), r*np.sin(theta), 0]))
    for angle, dist in zip(angles, ge_distances[1:]):
        if randomize_heading:
            theta += rng.choice([-1,1])*angle
        else:
            if np.dot(np.array([np.cos(theta + angle), np.sin(theta + angle), 0]),
                      point_x[0]) \
                          < \
               np.dot(np.array([np.cos(theta - angle), np.sin(theta - angle), 0]),
                      point_x[0]):
                theta -= angle
            else:
                theta += angle
        point_x.append(point_x[-1] + dist*np.array([np.cos(theta), np.sin(theta), 0]))

    points = []
    for i, point_index in enumerate(permutation):
        p = event.points[point_index]
        points.append(Interaction(point_x[i], p.e, ts=p.ts, crystal_no=p.crystal_no,
                                     seg_no=p.seg_no, event_no=p.event_no,
                                     interaction_id=p.interaction_id,
                                     interaction_type=p.interaction_type))
    return points

def flatten_event(event:Event, clusters:dict, correct_air_gap:bool=True,
                  random_seed:int = None, **kwargs) -> Event:
    """Flatten all of the clusters in an event. This creates a new event."""
    new_points = [0]*len(event.hit_points)
    for cluster in clusters.values():
        points = flatten_cluster(event, cluster, rng=np.random.RandomState(random_seed), **kwargs)
        for p, i in zip(points, cluster):
            new_points[i-1] = p
    new_event = Event(event.id, new_points, ground_truth=event.ground_truth, flat=True)
    if correct_air_gap:
        air_correction(new_event, clusters)
    return new_event

def air_correction_cluster(event:Event, cluster:list, debug:bool=False):
    """
    Convert all distances to germanium distances by extending any distances
    overlapping the detector center. This assumes that the distances between
    interactions are the desired germanium distances.

    This function mutates the event in place.

    TODO - fix for first point is not implemented, and is not likely needed,
    but could be added
    """
    distances = event.distance_perm(cluster)[1:]
    positions = event.point_matrix[list(cluster)]
    if debug:
        print(f'Distances to convert to germanium distances:\n{distances}')
        print(f'Original positions:\n{positions}')
    for i, (d, p1, p2) in enumerate(zip(distances, positions[:-1], positions[1:])):
        dg = ge_distance(np.vstack((p1,p2)))
        if ~np.isclose(dg, d):
            extra = germanium_extension(p1, p2)
            if debug:
                print(f'Found difference in euclidean and germanium distance: {d - dg}')
                print(extra)
            for j in range(i + 1,len(positions)):
                positions[j] += extra
    if debug:
        print(f'New positions:\n{positions}')
        inds = perm_to_transition(range(positions.shape[0]), D=2)
        print("New germanium distances:")
        print(squareform(ge_distance(positions))[inds])
    event.flush_position_caches()
    for i, p in zip(cluster, positions):
        event.points[i].x = p
    if debug:
        print(event.point_matrix[list(cluster)])

def air_correction(event:Event, clusters:Dict, debug:bool=False):
    """
    # Correct for air gaps in clusters

    Some augmented clusters cross the center of the detector incorrectly and
    have interactions inside the hollow detector center. This function extended
    the distances between interactions such that the distance is converted to a
    distance through the germanium. Applying this twice will increase all
    adjusted lengths again by the same amount.

    The original event is mutated in place.
    """
    for _, cluster in clusters.items():
        air_correction_cluster(event, cluster, debug=debug)

def augment_remove_head(event: Event, clusters: Dict, n: int = 1):
    """
    Remove the first n interactions from the provided clusters

    We reposition the following interactions as if the nth interaction was the
    target (origin)

    Does not perform the air-correction for passage through the center of the
    detector. (If a scatter crosses the center before or after the augmentation,
    it's distance is not corrected)

    TODO - add changes for air correction
    """
    event_copy = event.copy()
    removal_indices = []
    radius = event.distance[0,1] - event.ge_distance[0,1]
    for cluster_index in clusters:
        if n < len(clusters[cluster_index]):
            pn = event.points[clusters[cluster_index][n-1]].x  # want to rotate about this point
            pnp1 = event.points[clusters[cluster_index][n]].x
            theta = event.theta_act_perm([clusters[cluster_index][n-1],clusters[cluster_index][n]])
            axis = np.cross(np.array(pnp1) - pn, np.array(pn))
            axis = axis / np.linalg.norm(axis)  # normalize the rotation vector first

            rot = Rotation.from_rotvec(theta * axis)

            dist = event.ge_distance[clusters[cluster_index][n-1], clusters[cluster_index][n]]

            def apply_rotation(vec, center, rotation, dist):
                return rotation.apply(vec-center) + (center/np.linalg.norm(center))*(radius + dist)

            for index in clusters[cluster_index]:
                p = event.points[index]
                new_pos = apply_rotation(p.x, pn, rot, dist)
                event_copy.points[index] = p.update_position(new_pos)

            removal_indices.extend(clusters[cluster_index][:n])
    return remove_interactions(event_copy, removal_indices, clusters)

def random_rotation_about_origin(event:Event, clusters:Dict):
    """
    Rotate clusters randomly about the origin (doppler effect is messed up)

    TODO - check code
    """
    for cluster in clusters.values():
        theta = np.random.uniform()*np.pi
        axis = np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()])
        axis = axis/np.linalg.norm(axis)
        rot = Rotation.from_rotvec(theta * axis)
        for j in cluster:
            p = event.points[j]
            event.points[j] = p.update_position(rot.apply(p.x))
    return event, clusters

def random_chain_rotation(event, clusters):
    """
    Randomly rotate points about interaction axes.

    Maintains angles and distance, but not orientation (polarization is messed up)

    TODO - check code
    """

    def apply_rotation(vec, center, rotation):
        return rotation.apply(vec-center) + center

    for cluster in clusters.values():
        for i, index in enumerate(cluster):
            pn = event.points[index].x
            if i == 0:
                axis = pn - event.points[0].x
            else:
                axis = pn - event.points[cluster[i-1]].x
            theta = np.random.uniform()*np.pi
            rot = Rotation.from_rotvec(theta * axis)

            for j in cluster[i + 1:]:
                p = event.points[j]
                event.points[j] = p.update_position(apply_rotation(event.points[j].x, pn, rot))

    return event, clusters
