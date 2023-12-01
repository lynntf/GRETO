"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Cluster tools
"""
from __future__ import annotations
from copy import deepcopy
from itertools import product
from typing import Dict, Hashable, List, Tuple, Union

import numpy as np
from scipy.cluster.hierarchy import fcluster, fclusterdata
from scipy.spatial.distance import pdist, squareform

from .event_class import Event
from .fom_tools import cluster_FOM
from .geometry import ge_distance
from .interaction_class import Interaction


def remove_interactions(event:Event, removal_indices:List[int],
                        clusters:Dict[Hashable,List[int]]=None,
                        keep_empties:bool=False) -> Event:
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
            if len(new_clusters[cluster_idx]) == 0 and not keep_empties:
                del new_clusters[cluster_idx]

        return Event(event.id, new_points), new_clusters
    return Event(event.id, new_points)

def remove_zero_energy_interactions(event:Event,
                                    clusters:Dict[Hashable,int]=None,
                                    energy_threshold:float = 0.,
                                    keep_empties:bool = False) -> Event:
    """
    Remove zero energies for a single event
    """
    removal_indices = []
    for i, point in enumerate(event.hit_points):
        if point.e <= energy_threshold:
            removal_indices.append(i + 1)
    if clusters is not None:
        return remove_interactions(event, removal_indices, clusters,
                                   keep_empties=keep_empties)
    return remove_interactions(event, removal_indices)

def pack_interactions(event: Event,
                      packing_distance: float = 0.6,
                      clusters: dict = None,
                      keep_duplicates: bool = False
                      ) -> Union[Event, Tuple[Event, Dict]]:
    """
    Combine interactions that are too close (would be indistinguishable to the
    detector array).

    Default distance is 5 mm.

    Cluster labels can also be adjusted, but this cannot provide completely
    correct values if two different clusters become connected. If the same
    interaction is in multiple clusters, this will keep both.

    Args:
        packing_distance (float): distance to pack (default is in cm)
        clusters (dict): clusters for the event (optional, used to maintain
        clusters after packing)
        keep_duplicates (bool): keep the interaction order in each cluster even
        if this allows for repeated positions If this is False, any duplicates
        are discarded (order is kept otherwise)
    """
    if len(event.hit_points) <= 1:
        if clusters is not None:
            return deepcopy(event), deepcopy(clusters)
        return deepcopy(event)
    # Find the points that would join together; given a point, indicates
    # the cluster it should be in
    packing_clusters = fclusterdata([p.x for p in event.hit_points], \
        packing_distance, criterion='distance')
    fixed_order = {}
    order = 1
    for ii, index in enumerate(packing_clusters):
        if index not in fixed_order:
            fixed_order[index] = order
            packing_clusters[ii] = order
            order += 1
        else:
            packing_clusters[ii] = fixed_order[index]

    # Correct the indices in the clusters
    used = set()
    unique = set()
    if clusters is not None:
        new_clusters = {}
        # Loop over true clusters
        for key in clusters.keys():
            if keep_duplicates:
                unique = set()
            # Begin with empty cluster
            new_clusters[key] = []
            for index in clusters[key]:
                if index not in used and packing_clusters[index - 1] not in unique:
                    new_clusters[key].append(packing_clusters[index - 1])
                    used.add(index)
                    unique.add(packing_clusters[index - 1])
            if len(new_clusters[key]) == 0:
                del new_clusters[key]

    # Correct the points positions and energies, position is weighted by deposited energy
    points = []
    for cluster in set(packing_clusters):
        esum = 0
        interaction_type = 0
        position = np.array([0.,0.,0.])
        for i, p in enumerate(packing_clusters):
            if p == cluster:
                esum += event.hit_points[i].e
                position += event.hit_points[i].e*event.hit_points[i].x
                if event.hit_points[i].type is not None:
                    interaction_type = max(interaction_type, event.hit_points[i].type)
        points.append(Interaction(position/esum, esum,
                                    interaction_type=interaction_type))

    if clusters is not None:
        return Event(event.id, points, flat=event.flat, detector=event.detector_config), new_clusters
    return Event(event.id, points, flat=event.flat, detector=event.detector_config)

def apply_agata_error_model(event:Event, seed:int = None) -> Event:
    """
    Applies the AGATA error model to the positions and energies.
    This applies a gaussian error which is dependent on the energy
    deposited to both position and error.

    2.3458*variance is full-width at half-maximum for a Gaussian.

    Siciliano2021: pos_std = w0 + w1 * sqrt(100 keV / p.e)
    with w0 = 2.70(17) mm and w1 = 6.2(4) mm

    TODO - add support for subsets?
    """
    rng = np.random.RandomState(seed=seed)
    hit_points = deepcopy(event.hit_points)
    for p in hit_points:
        pos_std = (0.5 * np.sqrt(0.1/p.e)) / 2.3548
        p.x += rng.normal(0, pos_std, 3)
        # Divide by 1000 to convert to MeV
        e_std = np.sqrt(1 + 3.7 * p.e) / 2.3548 / 1000
        p.e += e_std * rng.normal(0, 1)
        # Minimum possible value is 0
        p.e = max(p.e, 0)
        # p.e = abs(p.e)
    return Event(event.id, hit_points, ground_truth=event.ground_truth,
                 flat=event.flat, detector=event.detector_config)

def apply_error_model(event:Event, seed:int = None) -> Event:
    """
    Applies the AGATA error model to the positions and energies.
    This applies a gaussian error which is dependent on the energy
    deposited to both position and error.

    Here the error in position is given by precomputed position uncertainty

    TODO - add support for subsets?
    """
    rng = np.random.RandomState(seed=seed)
    hit_points = deepcopy(event.hit_points)
    for i, p in enumerate(hit_points):
        pos_std = event.position_uncertainty[i + 1]
        p.x += rng.normal(0, pos_std, 3)
        # Divide by 1000 to convert to MeV
        p.e += event.energy_uncertainty[i + 1] * np.random.normal(0, 1)
        # Minimum possible value is 0
        p.e = max(p.e, 0)
    return Event(event.id, hit_points)

def cluster_linkage(event:Event, alpha:float = np.deg2rad(10.),
                    alpha_degrees:float = None,
                    max_clusters:int=30,
                    **linkage_kwargs) -> Dict[int,int]:
    """
    Clusters based on the linkage for the event
    TODO - add AGATA alpha
    TODO - add AGATA clustering method
    """
    if len(event.hit_points) == 0:
        return {}

    if len(event.hit_points) == 1:
        return {1: [1]}

    if alpha_degrees is not None:
        alpha = np.deg2rad(alpha_degrees)

    Z = event.linkage_array(**linkage_kwargs)

    # First try clustering on distance
    cluster_assignments = fcluster(Z, t=alpha, criterion='distance')
    # If too many, cluster based on max clusters
    if cluster_assignments.max() > max_clusters:
        cluster_assignments = fcluster(Z, t=max_clusters,
                                        criterion='maxclust')
    clusters = {}
    for i in range(1, max(cluster_assignments)+1):
        clusters[i] = list((cluster_assignments == i).nonzero()[0] + 1)
    return sort_clusters_by_min_index(clusters)

def sort_clusters_by_min_index(clusters:Dict[int,int]) -> Dict[int,int]:
    """Sort the clusters by their minimum index"""
    return dict(enumerate(sorted(clusters.values(), key=min)))

def cluster_summary(event:Event,
                    clusters:Dict[Hashable,int],
                    **FOM_kwargs):
    """
    Given a clustering of the points, write out a summary of statistics
    related to the clustering.
    Including:
        Energy of clusters
        FOM of clusters
        Points included in clusters
    """
    FOMs = cluster_FOM(event, clusters, **FOM_kwargs)
    summary = f"Event {event.id}\n"
    for i, cluster in clusters.items():
        summary += f"Cluster {i}:\n"
        summary += f"    Energy: {sum(event.points[j].e for j in cluster) : 4.4f}\n"
        summary += f"    FOM:    {FOMs[i] : 4.4f}\n"
        summary += "    Points (in order):\n"
        for j, pt_idx in enumerate(cluster):
            pt = event.points[pt_idx]
            summary += f'        {j}, index {pt_idx}: pos=[{pt.x[0]: >6.2f},'+\
                f' {pt.x[1]: >6.2f}, {pt.x[2]: >6.2f}], energy={pt.e : 4.4f}\n'
        # summary  += '\n'
    return summary

def split_event(event:Event, clusters:Dict):
    """
    Split up an event by its clusters.
    May speed up computation on individual events.
    """
    new_events, new_clusters = [], []
    for i, cluster in clusters.items():
        new_events.append(Event(event.id, [event.points[j] for j in cluster]))
        new_clusters.append({i:tuple(range(1,len(cluster) + 1))})
    return new_events, new_clusters

def split_events(list_of_events:List[Event],
                 list_of_clusters:List[Dict]) -> Tuple[List[Event], List[Dict]]:
    """
    Split up many events by their clusters.
    May speed up computation on individual events.
    """
    new_events = []
    new_clusters = []
    for event, clusters in zip(list_of_events, list_of_clusters):
        # more_events, more_clusters = split_event(event, clusters)
        # new_events.extend(more_events)
        # new_clusters.extend(more_clusters)
        for i, cluster in clusters.items():
            new_events.append(Event(event.id, [event.points[j] for j in cluster]))
            new_clusters.append({i:list(range(1,len(cluster) + 1))})
    return new_events, new_clusters

def join_events(list_of_events_to_join:List[Event],
                list_of_clusters_to_combine:List[Dict[Hashable,List[int]]]) -> Tuple[Event, Dict]:
    """Combine events and clusters"""
    combined_points = []
    combined_clusters = {}
    current_cluster_indices = []
    current_point_index = 0
    for event, clusters in zip(list_of_events_to_join,
                               list_of_clusters_to_combine):
        combined_points.extend(event.hit_points)
        for cluster_index, cluster in clusters.items():
            if cluster_index not in current_cluster_indices:
                current_cluster_index = cluster_index
                current_cluster_indices.append(cluster_index)
            else:
                current_cluster_index = max(current_cluster_indices) + 1
                current_cluster_indices.append(current_cluster_index)
            combined_clusters[current_cluster_index] = [j + current_point_index for j in cluster]
            current_cluster_index += 1
        current_point_index += len(event.hit_points)
    return Event(list_of_events_to_join[0].id, combined_points), combined_clusters

#%% Cluster features

def cluster_centroid(event: Event, cluster: Tuple) -> np.ndarray:
    """
    Return the Euclidean centroid of the cluster.
    """
    return np.mean(event.point_matrix[np.array(cluster)], axis = 0)

def get_centroids(event: Event, clusters: Dict) -> np.ndarray:
    """
    Return the Euclidean centroids of the clusters.
    """
    centroids = np.zeros((len(clusters), event.points[0].x.shape[0]))
    for i, c1 in enumerate(clusters.values()):
        centroids[i] = cluster_centroid(event, tuple(c1))
    return centroids

def get_centroid_event(event: Event, clusters: Dict, method:str = 'centroid') -> Event:
    """
    Convert the clusters given to single points at their centroid (or barycenter).
    """
    if method == 'centroid':
        reduce_func = cluster_centroid
    elif method == 'barycenter':
        reduce_func = cluster_barycenter
    points = []
    for cluster in clusters.values():
        x = reduce_func(event, tuple(cluster))
        e = np.sum(event.energy_matrix[np.array(cluster)])
        ts = np.mean([event.points[i].ts for i in cluster], dtype = int)
        seg_no = np.mean([event.points[i].seg_no for i in cluster], dtype = int)
        crystal_no = np.mean([event.points[i].crystal_no for i in cluster], dtype = int)
        points.append(Interaction(x, e, ts = ts, crystal_no = crystal_no, seg_no = seg_no))
    return Event(event.id, points)

def cluster_barycenter(event: Event, cluster: Tuple) -> np.ndarray:
    """
    Return the Euclidean barycenter of the cluster using the deposited energy.
    """
    return np.sum(event.point_matrix[np.array(cluster)] * \
                  event.energy_matrix[np.array(cluster)][:, np.newaxis], axis = 0) / \
            np.sum(event.energy_matrix[np.array(cluster)], axis = 0)

def get_barycenters(event: Event, clusters: Dict) -> np.ndarray:
    """
    Return the Euclidean barycenters of the clusters using the deposited energy.
    """
    barycenters = np.zeros((len(clusters), event.points[0].x.shape[0]))
    for i, c1 in enumerate(clusters.values()):
        barycenters[i] = cluster_barycenter(event, tuple(c1))
    return barycenters

def cluster_pdist(event: Event, clusters: Dict,
                  method:str = 'single', metric:str = 'germanium'):
    """
    Get pairwise distance between clusters using the prescribed method and metric.
    """
    if len(event.hit_points) < 2:
        return np.array([[0.]])
    if method in ['centroid', 'barycenter']:
        if method == 'centroid':
            centers = get_centroids(event, clusters)
        else:
            centers = get_barycenters(event, clusters)
        if metric == 'euclidean':
            return squareform(pdist(centers, metric='euclidean'))
        elif metric == 'germanium':
            return squareform(ge_distance(centers))
        elif metric == 'angle':
            return squareform(np.arccos(1 - pdist(centers, metric='cosine')))

    pairwise_distance = np.zeros((len(clusters), len(clusters)))

    if metric == 'euclidean':
        distances = event.distance
    elif metric == 'germanium':
        distances = event.ge_distance
    elif metric == 'angle':
        distances = np.pad(event.angle_distance, ((1,0), (1,0)))

    if method == 'asym':
        for (i, c1), (j, c2) in product(enumerate(clusters.values()), enumerate(clusters.values())):
            # we could define a non-zero distance from tail to head, but this makes no sense
            if i != j:
                pairwise_distance[i,j] = distances[c1[-1],c2[0]]
        return pairwise_distance

    if method == 'head_tail':
        for (i, c1), (j, c2) in product(enumerate(clusters.values()), enumerate(clusters.values())):
            # we could define a non-zero distance from tail to head, but this makes no sense
            if i > j:
                pairwise_distance[i,j] = min(distances[c1[-1],c2[0]], distances[c1[0],c2[-1]])
                pairwise_distance[j,i] = pairwise_distance[i,j]
        return pairwise_distance

    if method in ['single', 'complete', 'average']:
        if method == 'single':
            reduce_func = np.min
        elif method == 'complete':
            reduce_func = np.max
        elif method == 'average':
            reduce_func = np.mean

        for (i, c1), (j, c2) in product(enumerate(clusters.values()), enumerate(clusters.values())):
            if i > j:
                pairwise_distance[i,j] = reduce_func(
                    distances[np.array(c1)[:, np.newaxis],np.array(c2)]
                    )
                pairwise_distance[j,i] = pairwise_distance[i,j]
        return pairwise_distance

# What sorts of properties do we want to measure? What sorts of features can we
# get for individual clusters? For single interactions? Should those features
# come from the individual cluster alone, or are they defined relative to other
# clusters?

# Lets say that we want both cluster specific features. In what context should
# they be used? For clustering, relative cluster properties make sense. For
# suppression, they do not. At the suppression step, the cluster is completely
# independent from anything else. At the clustering and ordering steps where the
# cluster is still mutable, relative features are important. We can of course
# assume that relative features are not important for the ordering step without
# much loss since the ordering is mostly independent of other clusters (it is
# completely independent when the cluster contains the correct interactions).

class cluster_properties:
    """
    Properties class for a cluster.

    TODO - convert to dataclass?
    """
    def __init__(self, event:Event, cluster:Tuple, start_point:int = 0):
        self.cluster = cluster
        self.centroid = cluster_centroid(event, cluster)
        self.barycenter = cluster_barycenter(event, cluster)
        points_x = event.point_matrix[[start_point] + list(cluster)]
        self.vectors = points_x[1:] - points_x[:-1]
        length_vector = np.sum(self.vectors, axis=0)
        self.directions = self.vectors/np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.esum = event.energy_sum(cluster)

        if len(cluster) > 1:
            self.average_dir = np.mean(self.directions, axis=0)
            self.length_dir = length_vector/np.linalg.norm(length_vector)
            self.centroid_vectors = points_x[1:] - self.centroid
            lengths = np.abs(np.dot(self.centroid_vectors, self.length_dir))
            self.length = np.mean(lengths)
            length_vectors = lengths[:,None] * self.length_dir[None,:]
            width_vectors = self.centroid_vectors - length_vectors
            widths = np.linalg.norm(width_vectors, axis=-1)
            self.width = np.mean(widths)
            self.first_energy_ratio = event.energy_matrix[cluster[0]]/self.esum
            self.final_energy_ratio = event.energy_matrix[cluster[-2]]/\
                (event.energy_matrix[cluster[-2]] + event.energy_matrix[cluster[-1]])

            self.first_is_not_largest = np.all(
                event.energy_matrix[list(cluster)][0] < event.energy_matrix[list(cluster)][1:]
            )
            self.first_is_not_closest = np.all(
                event.spherical_point_matrix[list(cluster)][:,0][0] > event.spherical_point_matrix[list(cluster)][:,0][1:]
            )

            tangos = event.tango_estimates_perm(cluster, start_point=start_point)
            tangos_sigma = event.tango_estimates_sigma_perm(cluster,start_point=start_point)
            self.tango_variance = np.var(tangos)
            self.tango_v_variance = 1/np.sum(1/tangos_sigma**2)
            self.tango_sigma = np.sqrt(np.var(tangos))
            self.tango_v_sigma = np.sqrt(1/np.sum(1/tangos_sigma**2))
        else:
            self.average_dir = self.directions[0]
            self.length_dir = self.directions[0]
            self.length = event.position_uncertainty[cluster[0]]
            self.width = event.position_uncertainty[cluster[0]]
            self.first_energy_ratio = 1.
            self.final_energy_ratio = 1.
            self.first_is_not_largest = False
            self.first_is_not_closest = False
            self.tango_variance = 0
            self.tango_v_variance = 0
            self.tango_sigma = 0
            self.tango_v_sigma = 0

        self.aspect_ratio = self.length/self.width

        self.features = {
            "esum" : self.esum,
            "n" : len(self.cluster),
            "centroid_r" : np.linalg.norm(self.centroid),
            "average_r" : np.mean(event.distance[0,list(cluster)]),
            "first_r" : event.distance[0, cluster[0]],
            "final_r" : event.distance[0,cluster[-1]],
            "length" : self.length,
            "width" : self.width,
            "aspect_ratio" : self.aspect_ratio,
            "first_energy_ratio" : self.first_energy_ratio,
            "final_energy_ratio" : self.final_energy_ratio,
            "first_is_not_largest" : self.first_is_not_largest,
            "first_is_not_closest": self.first_is_not_closest,
            "tango_variance": self.tango_variance,
            "tango_v_variance": self.tango_v_variance,
            "tango_sigma": self.tango_sigma,
            "tango_v_sigma": self.tango_v_sigma,
        }
    def __repr__(self) -> str:
        s = "{\n"
        for key, value in self.features.items():
            s += f'"{key}" : {value},\n'
        s += "}"
        return s
    def __str__(self) -> str:
        pass

# class cluster_contextual_properties:
#     def __init__(self, event:Event, clusters:Dict):
#         return

#%% Old cluster features

class cluster_calcs():
    """
    Class to store and call precomputed distances and features
    """
    def __init__(self, event:Event):
        self.d_euc   = squareform(event.distance)
        self.d_ge    = squareform(event.ge_distance)
        self.d_angle = squareform(event.angle_distance)
        self.m = len(event.hit_points)
        self.event = event
    def euc(self, i, j):
        """
        Return precalculated Euclidean distance between points i and j
        """
        if i == j:
            return 0
        if i > j:
            j, i = i, j
        return self.d_euc[(self.m + 1) * i + j - ((i + 2) * (i + 1)) // 2]
    def ge(self, i, j):
        """
        Return precalculated Germanium distance between points i and j
        """
        if i == j:
            return 0
        if i > j:
            j, i = i, j
        return self.d_ge[(self.m + 1) * i + j - ((i + 2) * (i + 1)) // 2]
    def angle(self, i, j):
        """
        Return precalculated angular distance between points i and j
        """
        if i == j:
            return 0
        if i > j:
            j, i = i, j
        return self.d_angle[(self.m + 1) * i + j - ((i + 2) * (i + 1)) // 2]
    def cluster_linkage(self, cluster1, cluster2, method='single', metric='euclidean'):
        """
        Get linkage distances between cluster1 and cluster2.

        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        """
        if metric == 'euclidean':
            metric_method = self.euc
        elif metric == 'germanium':
            metric_method = self.ge
        elif metric == 'angle':
            metric_method = self.angle

        if method == 'single':
            d = np.inf
        elif method == 'complete':
            d = 0
        elif method == 'average':
            d = 0

        if method in ['single', 'complete', 'average']:
            for p1, p2 in product(cluster1, cluster2):
                dnew = metric_method(p1, p2)
                if method == 'single':
                    d = min(d, dnew)
                if method == 'complete':
                    d = max(d, dnew)
                if method == 'average':
                    d += dnew/(len(cluster1)*len(cluster2))

        if method == 'centroid':
            if metric == 'euclidean':
                d = pdist([self.cluster_centroid(cluster1), self.cluster_centroid(cluster2)],
                          'euclidean')[0]
            if metric == 'angle':
                d = np.arccos(1 - pdist([self.cluster_centroid(cluster1),
                                         self.cluster_centroid(cluster2)],
                                        'cosine')[0])
            if metric == 'germanium':
                # TODO - This is not entirely clear. Centroids might be computed differently here...
                d = ge_distance(np.array([self.cluster_centroid(cluster1),
                                self.cluster_centroid(cluster2)]))

        return d

    def cluster_centroid(self, cluster):
        """
        Return the Euclidean centroid of the cluster
        """
        return np.mean([self.event.points[i].x for i in cluster], axis = 0)

    def cluster_properties(self, cluster, sigma_thet = 0.2, label=''):
        """
        Get some properties from the cluster including shape, direction, radial location
        """
        centroid = self.cluster_centroid(cluster)

        vecs = [self.event.points[i].x - self.event.points[j].x
                for i,j in zip(cluster, [0] + cluster)]
        dirs = [vec / np.linalg.norm(vec) for vec in vecs]
        e_sum = sum(self.event.points[i].e for i in cluster)

        if len(cluster) > 1:
            average_dir = np.mean(dirs[1:], axis=0)
            length_vec = self.event.points[cluster[-1]].x - self.event.points[cluster[0]].x
            length_dir = length_vec/np.linalg.norm(length_vec)
            centroid_vecs = np.array([self.event.points[i].x - centroid for i in cluster])
            lengths = np.array([vec @ length_dir
                                for vec in centroid_vecs])
            length = 2*np.sqrt(np.var(lengths))
            length_vecs = length*length_dir
            width_vecs = centroid_vecs - length_vecs
            widths = np.array([np.linalg.norm(vec)
                               for vec in width_vecs])
            width = 2*np.sqrt(np.var(widths))

            first_e_diff = self.event.points[cluster[1]].e - self.event.points[cluster[0]].e
            final_e_diff = self.event.points[cluster[-2]].e - self.event.points[cluster[-1]].e

            tango_origin = self.event.estimate_start_energy(cluster, normalize_by_sigma=True)
            tango = self.event.estimate_start_energy(cluster,
                                                     normalize_by_sigma=True,
                                                     start_point_index=None)

        else:
            average_dir = dirs[0]
            length_dir = dirs[0]
            length = sigma_thet
            width = sigma_thet
            first_e_diff = 0
            final_e_diff = 0
            tango_origin = 0
            tango = 0

        properties = {f'{label}n': len(cluster),
                  f'{label}centroid_r': np.linalg.norm(centroid),
                  f'{label}average_r' : np.mean([self.euc(0,i) for i in cluster]),
                  f'{label}first_r': self.euc(0, cluster[0]),
                  f'{label}final_r': self.euc(0, cluster[-1]),
                  f'{label}length': length,
                  f'{label}width': width,
                  f'{label}first_dir': dirs[0],
                  f'{label}final_dir': dirs[-1],
                  f'{label}average_dir': average_dir,
                  f'{label}length_dir': length_dir,
                  f'{label}first_e_diff': first_e_diff,
                  f'{label}final_e_diff': final_e_diff,
                  f'{label}e_sum': e_sum,
                  f'{label}tango_origin': tango_origin,
                  f'{label}tango': tango}
        return properties

    def relative_cluster_properties(self, cluster1, cluster2):
        """
        Get the features between clusters 1 and clusters 2 (and their individual
        features)
        """
        c1p = self.cluster_properties(cluster1, label='1')
        c2p = self.cluster_properties(cluster2, label='2')
        join_vec = self.event.points[cluster1[-1]].x - self.event.points[cluster2[0]].x
        join_dir = join_vec/np.linalg.norm(join_vec)
        if np.linalg.norm(join_vec) == 0:
            print(cluster1, cluster2)

        # Estimated energy flowing into cluster2
        if len(cluster2) > 1:
            tango12 = self.event.estimate_start_energy([cluster1[-1]] + cluster2,
                                                       normalize_by_sigma=True,
                                                       start_point_index=None)
        elif len(cluster1) > 1:
            tango12 = self.event.estimate_start_energy(cluster1[-2:] + cluster2,
                                                       normalize_by_sigma=True,
                                                       start_point_index=None) - \
                                                           self.event.points[cluster1[-2]].e
        else:
            tango12 = self.event.estimate_start_energy(cluster1 + cluster2,
                                                       normalize_by_sigma=True,
                                                       start_point_index=0) - \
                                                           self.event.points[cluster1[-1]].e

        # Estimated total energy
        total_tango = self.event.estimate_start_energy(cluster1 + cluster2,
                                                       normalize_by_sigma=True,
                                                       start_point_index=0)

        relative_properties = {'join_d_first2': join_dir @ c2p['2first_dir'],
                               'join_d_final1': join_dir @ c1p['1final_dir'],
                               'join_d_average1': join_dir @ c1p['1average_dir'],
                               'join_d_average2': join_dir @ c2p['2average_dir'],
                               'join_d_length1': join_dir @ c1p['1length_dir'],
                               'join_d_length2': join_dir @ c2p['2length_dir'],
                               'length1_d_length2': c1p['1length_dir'] @ c2p['2length_dir'],
                               'length1_d_first2': c1p['1length_dir'] @ c2p['2first_dir'],
                               'final1_d_length2': c1p['1final_dir'] @ c2p['2length_dir'],
                               'final1_d_first2': c1p['1final_dir'] @ c2p['2first_dir'],
                               'tango_12': tango12,
                               'single_linkage_angle': self.cluster_linkage(cluster1, cluster2, metric='angle', method='single'),
                               'complete_linkage_angle': self.cluster_linkage(cluster1, cluster2, metric='angle', method='complete'),
                               'average_linkage_angle': self.cluster_linkage(cluster1, cluster2, metric='angle', method='average'),
                               'centroid_linkage_angle': self.cluster_linkage(cluster1, cluster2, metric='angle', method='centroid'),
                               'single_linkage_ge': self.cluster_linkage(cluster1, cluster2, metric='germanium', method='single'),
                               'complete_linkage_ge': self.cluster_linkage(cluster1, cluster2, metric='germanium', method='complete'),
                               'average_linkage_ge': self.cluster_linkage(cluster1, cluster2, metric='germanium', method='average'),
                               'centroid_linkage_ge': self.cluster_linkage(cluster1, cluster2, metric='germanium', method='centroid'),
                               'tango_12 - 2e_sum': np.abs(tango12 - c2p['2e_sum']), # If cluster2 is a complete tail, the energy coming in is equal to the energy present
                               '1tango_origin - tango12 - 1e_sum': np.abs(c1p['1tango_origin'] - (tango12 + c1p['1e_sum'])), # If cluster1 is a complete head, the energy between entry and exit is equal to the energy present
                               '1final_r - 2first_r': c1p['1final_r'] - c2p['2first_r'],
                               '1tango_origin - 1e_sum': np.abs(c1p['1tango_origin'] - c1p['1e_sum']), # If cluster1 is complete, the sum of energies and TANGO estimate will be close
                               '1e_sum - 1tango - 1first_e': np.abs(c1p['1e_sum'] - (c1p['1tango'] + self.event.points[cluster1[0]].e)), # If cluster1 is complete, check TANGO and first energy
                               '2tango_origin - 2e_sum': np.abs(c2p['2tango_origin'] - c2p['2e_sum']), # If cluster2 is complete, the sum of energies and TANGO estimate will be close
                               '2e_sum - 2tango - 2first_e': np.abs(c2p['2e_sum'] - (c2p['2tango'] + self.event.points[cluster2[0]].e)), # If cluster1 is complete, check TANGO and first energy
                               'total_tango - e_sum': np.abs(total_tango - (c1p['1e_sum'] + c2p['2e_sum'])),
                               'n': c1p['1n'] + c2p['2n'],
                               'angle_FOM': self.event.FOM(cluster1 + cluster2, fom_method='angle', compton_penalty=None),
                               'angle_FOM_varcos': self.event.FOM(cluster1 + cluster2, fom_method='angle', compton_penalty=None, include_sigma_cos=True),
                               'angle_FOM_TANGO': self.event.FOM(cluster1 + cluster2, fom_method='angle', compton_penalty=None, start_energy=total_tango),
                               'angle_FOM_TANGO_varcos': self.event.FOM(cluster1 + cluster2, fom_method='angle', compton_penalty=None, start_energy=total_tango, include_sigma_cos=True),
                               'local_FOM': self.event.FOM(cluster1 + cluster2, fom_method='local'),
                               'local_FOM_TANGO': self.event.FOM(cluster1 + cluster2, fom_method='local', start_energy=total_tango),
                               'agata_FOM': self.event.FOM(cluster1 + cluster2, fom_method='agata'),
                               'agata_FOM_TANGO': self.event.FOM(cluster1 + cluster2, fom_method='agata', start_energy=total_tango),
#                                '1tango'
#                                'FOM' we want something like a FOM, comparing cosine values or energy values
#                                FOM of new joined cluster
#                                Something like the Compton scattering formula needs to be included
                            }
        del c1p['1first_dir'], c1p['1final_dir'], c1p['1average_dir'], c1p['1length_dir'], c1p['1e_sum']
        del c2p['2first_dir'], c2p['2final_dir'], c2p['2average_dir'], c2p['2length_dir'], c2p['2e_sum']
        return relative_properties | c1p | c2p


def cluster_using_classifier(event, points:list[int]=None, prediction_func=None, cutoff:float=1.,
                             debug:bool=False, return_data:bool=False,
                             clusters:dict[int,list[int]]=None, balance_data:bool=False,
                             use_oracle:bool=False):
    """
    Apply the classifier prediction_func to the given event.

    If returning data, the true clusters need to be provided. Balancing the data
    will add in duplicates of cluster data (not duplicating disjoint data).

    Uses numpy arrays to store scores. Uses the entire event assuming that
    points have indices 1..n
    """
    if prediction_func is None and not return_data:
        raise ValueError("Prediction function is needed to return data.")

    if points is None:
        new_clusters = [[i+1] for i in range(len(event.hit_points))] # start with singles
    else:
        new_clusters = [[i] for i in points]
    if len(new_clusters) <= 1: # Exit if there is nothing to cluster
        if return_data:
            return [], []
        return {1: new_clusters[0]}

    ed = cluster_calcs(event) # Initialize cluster calc object
    scores = np.zeros((len(new_clusters), len(new_clusters))) # Initialize score matrix
    first = True

    if return_data: # If returning data instead of clusters, initialize data variables
        X = []
        y = []

    multiplicity = 0 # Do not duplicate any data

    while len(new_clusters) > 1:
        if balance_data and return_data: # Duplicate data if requested
            multiplicity = len(new_clusters) - 2
        batch = [] # Perform predictions on a batch of values
        preds = []
        indices = [] # Indicate the indices for the batched values
        best_clusters = []
        if first: # Populate the scores for all pairs on the first pass
            for i, c1 in enumerate(new_clusters):
                for j, c2 in enumerate(new_clusters):
                    if i != j:
                        x = list(ed.relative_cluster_properties(c1,c2).values())
                        if not any(np.isnan(x)) and not any(np.isinf(x)):
                            batch.append(x)
                            if use_oracle:
                                preds.append(float(cluster_utils.join_validity(clusters, c1, c2)) + float(cluster_utils.end_validity(clusters, c1, c2)))
                            indices.append((i,j))
                            if return_data:
                                y_local = float(cluster_utils.join_validity(clusters, c1, c2)) + float(cluster_utils.end_validity(clusters, c1, c2))
                                X.append(x)
                                y.append(y_local)
                                if y_local > 0:
                                    for k in range(multiplicity):
                                        X.append(x)
                                        y.append(y_local)
                        else:
                            scores[i,j] = 0
            if prediction_func is None and not use_oracle:
                return X, y

        else: # Otherwise just populate the scores relative to the previously created cluster
            for i, c1 in enumerate(new_clusters):
                if i != overwrite_index:
                    c2 = new_clusters[overwrite_index]
                    x = list(ed.relative_cluster_properties(c1, c2).values())
                    if not any(np.isnan(x)) and not any(np.isinf(x)):
                        batch.append(x)
                        if use_oracle:
                            preds.append(float(cluster_utils.join_validity(clusters, c1, c2)) + float(cluster_utils.end_validity(clusters, c1, c2)))
                        indices.append((i,overwrite_index))
                        if return_data:
                            y_local = float(cluster_utils.join_validity(clusters, c1, c2))\
                                + float(cluster_utils.end_validity(clusters, c1, c2))
                            X.append(x)
                            y.append(y_local)
                            if y_local > 0:
                                for k in range(multiplicity):
                                    X.append(x)
                                    y.append(y_local)
                    else:
                        scores[i, overwrite_index] = 0

                    x = list(ed.relative_cluster_properties(c2, c1).values())
                    if not any(np.isnan(x)) and not any(np.isinf(x)):
                        batch.append(x)
                        if use_oracle:
                            preds.append(float(cluster_utils.join_validity(clusters, c2, c1)) + float(cluster_utils.end_validity(clusters, c2, c1)))
                        indices.append((overwrite_index,i))
                        if return_data:
                            y_local = float(cluster_utils.join_validity(clusters, c2, c1))\
                                + float(cluster_utils.end_validity(clusters, c2, c1))
                            X.append(x)
                            y.append(y_local)
                            if y_local > 0:
                                for k in range(multiplicity):
                                    X.append(x)
                                    y.append(y_local)
                    else:
                        scores[overwrite_index, i] = 0

        if not use_oracle:
            if len(batch) == 0:
                break
            preds = prediction_func(batch)
        for (i,j), pred in zip(indices, preds):
            scores[i,j] = pred
        if np.max(scores) > cutoff:
            if not use_oracle:
                b1, b2 = np.unravel_index( # pylint: disable=unbalanced-tuple-unpacking
                    np.argmax(scores, axis=None), scores.shape
                    )
            else:
                b1, b2 = np.unravel_index( # pylint: disable=unbalanced-tuple-unpacking
                    np.random.choice(np.flatnonzero(scores == np.max(scores))), scores.shape
                    )
            best_clusters = [new_clusters[b1], new_clusters[b2]]
            del new_clusters[max(b1, b2)]
            new_clusters[min(b1, b2)] = best_clusters[0] + best_clusters[1]
            if debug:
                print(f'Selected indices {b1} and {b2}: ')
                print(new_clusters)
            scores = np.delete(scores, max(b1, b2), 0)
            scores = np.delete(scores, max(b1, b2), 1)
            overwrite_index = min(b1, b2)
            first = False
        else:
            if return_data:
                return X, y
            return {(s+1): cluster for s, cluster in enumerate(new_clusters)}
    if return_data:
        return X, y
    return {(s+1): cluster for s, cluster in enumerate(new_clusters)}

#%% Packing and smearing

def pack_and_smear(
    event:Event,
    clusters:Dict = None,
    packing_distance: float = 0.6,
    energy_threshold:float=0.,
    use_agata_model:bool = True,
    seed:int = None) -> Union[Event, Tuple[Event, Dict]]:
    """
    Apply the packing and smearing methods to the event and clusters. Any
    interactions with energy less than the threshold value are deleted.
    """
    if clusters is None:
        packed_event = pack_interactions(event, packing_distance = packing_distance)
        if use_agata_model:
            packed_and_smeared_event = apply_agata_error_model(packed_event)
        else:
            packed_and_smeared_event = apply_error_model(packed_event)
        threshold_event = remove_zero_energy_interactions(
            packed_and_smeared_event, energy_threshold=energy_threshold)
        return threshold_event
        # return packed_and_smeared_event

    packed_event, packed_clusters = pack_interactions(event, clusters = clusters,
                                                            packing_distance = packing_distance)
    if use_agata_model:
        packed_and_smeared_event = apply_agata_error_model(packed_event,seed=seed)
    else:
        packed_and_smeared_event = apply_error_model(packed_event,seed=seed)
    threshold_event, threshold_clusters = remove_zero_energy_interactions(
        packed_and_smeared_event,
        packed_clusters, energy_threshold=energy_threshold)
    return threshold_event, threshold_clusters
    # return packed_and_smeared_event, packed_clusters

def pack_and_smear_list(list_of_events:List[Event],
                        list_of_clusters:List[Dict] = None,
                        packing_distance: float = 0.6,
                        energy_threshold=0.005,
                        use_agata_model:bool = True,
                        seed:int = None):
    """
    Apply the packing and smearing methods to a list of events and clusters
    """
    if list_of_clusters is None:
        ps_events = []
        for event in list_of_events:
            ps_events.append(pack_and_smear(event,
                                            packing_distance=packing_distance,
                                            energy_threshold=energy_threshold,
                                            use_agata_model=use_agata_model,
                                            seed=seed))
        return ps_events

    ps_events = []
    ps_clusters = []
    for event, clusters in zip(list_of_events, list_of_clusters):
        single_ps_event, single_ps_clusters = pack_and_smear(event,
                                                             clusters=clusters,
                                                             packing_distance=packing_distance,
                                                             energy_threshold=energy_threshold,
                                                             seed=seed)
        ps_events.append(single_ps_event)
        ps_clusters.append(single_ps_clusters)
    return ps_events, ps_clusters

def pack_list(list_of_events:List[Event],
              list_of_clusters:List[Dict] = None,
              packing_distance: float = 0.6):
    """
    Apply the packing and smearing methods to a list of events and clusters
    """
    if list_of_clusters is None:
        p_events = []
        for event in list_of_events:
            p_events.append(pack_interactions(event,
                                               packing_distance=packing_distance))
        return p_events

    p_events = []
    p_clusters = []
    for event, clusters in zip(list_of_events, list_of_clusters):
        single_p_event, single_p_clusters = pack_interactions(event,
                                                                clusters=clusters,
                                                                packing_distance=packing_distance)
        p_events.append(single_p_event)
        p_clusters.append(single_p_clusters)
    return p_events, p_clusters
