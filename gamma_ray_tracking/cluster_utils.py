"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Utilities for clustering gamma ray interactions and evaluating clusters.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Dict, Iterable, List

import numpy as np
import pyomo.environ as pyenv

from gamma_ray_tracking.event_class import Event


class Clustering(dict):
    """
    A class which encodes a clustering. Internally it contains a disjoint
    set data structure to enable easy access to the cluster
    """

    def __init__(self, starting_clusters):
        """
        starting_clusters is a mapping from cluster index to elements in
        the cluster.
        """
        dict.__init__(self, starting_clusters)
        # dictionary of parents where each key is mapped to its parent
        # start with everything maps to itself
        self.parent_map = {
            elem: cluster[0]
            for cluster in starting_clusters.values()
            for elem in cluster
        }
        # dictionary of root: size_of_tree
        # also serves as something to store the roots
        self.root_size_map = {
            elem: len(cluster)
            for cluster in starting_clusters.values()
            for elem in cluster
        }
        self.classes = starting_clusters
        self.size = sum(len(cluster) for cluster in starting_clusters.values())
        self.num_clusters = len(starting_clusters)

    ### METHODS ###
    def add_singleton(self, new_value):
        """
        Add a new value as its own equiv class
        """
        # Check first that it is not already in a member
        error_msg = (
            f"Equivalence.add_singleton: {new_value} is already "
            + "in the equiv class!"
        )
        assert new_value not in self.parent_map, error_msg

        self.parent_map[new_value] = new_value
        self.root_size_map[new_value] = 1

    def merge_classes_of(self, a, b):
        """
        Merge the clusters of a and b together
        by setting each parent's map to point to the same root.
        """
        assert a in self.parent_map, "Value " + str(a) + " does not exist."
        assert b in self.parent_map, "Value " + str(b) + " does not exist."

        # get the roots while compressing the tree
        root_of_a = self.compress_to_root(a)
        root_of_b = self.compress_to_root(b)

        if root_of_a == root_of_b:
            return  # they are already in the same equivalence class

        # find the "big" root and the "small" root
        if self.root_size_map[root_of_a] < self.root_size_map[root_of_b]:
            small_root, big_root = root_of_a, root_of_b
        else:
            small_root, big_root = root_of_b, root_of_a

        # now we change the root of the smaller size map to the bigger one
        # then we update the size of the new equiv class and then remove
        # the smaller one as a root
        self.parent_map[small_root] = big_root
        self.root_size_map[big_root] += self.root_size_map[small_root]

        # remove the small_root from the roots dict
        del self.root_size_map[small_root]

    def merge_set(self, set_to_merge):
        """
        Given a set of elements in Clustering, merge them together
        """
        for a in set_to_merge:
            assert a in self.parent_map, "Value " + str(a) + " does not exist."
        one_elem = next(iter(set_to_merge))
        for a in set_to_merge:
            self.merge_classes_of(one_elem, a)

    ### QUERIES ###
    def in_same_class(self, a, b) -> bool:
        """
        return a bool indicating if a and b are in the same classes
        """
        assert a in self.parent_map, str(a) + " does not exist."
        assert b in self.parent_map, str(b) + " does not exist."
        root_of_a = self.compress_to_root(a)
        root_of_b = self.compress_to_root(b)
        return root_of_a == root_of_b

    def __len__(self):
        """
        Returns the number of elements in the equiv class
        """
        return self.size

    def class_count(self):
        """
        return the number of unique equiv classes
        """
        return len(self.root_size_map)

    def get_class(self, elem):
        """Return the class of the elements"""
        return self.classes[self.representative(elem)]

    def non_trivial_classes(self):
        """
        This function returns a dictionary whose values are the equivalence
        classes and whose keys are the representatives of the classes.
        The dictionary will only contain classes which have more than one
        element.
        """
        return {
            rep: class_ for (rep, class_) in self.classes.items() if len(class_) > 1
        }

    def compress_to_root(self, elem):  # -> 'root of a':
        """
        Returns the root of elem, and on the way, set the roots of the
        parents of elem to be the root of elem
        """
        assert elem in self.parent_map, str(elem) + " does not exist."
        parents_of_elem = set()  # set to store the parents of elem
        curr_val = elem  # current value

        # We traverse up through the ancestors of elem and keep track
        # of the elements that we see so that we can compress the path
        while curr_val != self.parent_map[curr_val]:
            parents_of_elem.add(curr_val)
            curr_val = self.parent_map[curr_val]

        # curr_val is now the root
        # now we set all the parents of elem to the root
        for parent in parents_of_elem:
            self.parent_map[parent] = curr_val
        return curr_val

    def representative(self, elem):
        """
        Return the representative of elem's equivalence class.
        This performs path compression in the process of finding the
        representative.
        """
        return self.compress_to_root(elem)

    def cluster_idx_mapping(self):
        """
        Return an array mapping of self.size integers where the ith entry
        is the cluster that element i belongs too
        """
        mapping = np.zeros(self.size)
        for c_idx, elements in self.classes.items():
            for elem in elements:
                mapping[elem - 1] = c_idx
        return mapping

    def copy(self):
        """
        Return a deep copy of self
        """
        return deepcopy(self)


def get_permutation(instance):
    """Get the permutation for this instance"""
    perm = []
    for j in instance.J:
        for i in instance.I:
            if instance.z[i, j].value > 0.5:
                perm.append(i)
                break
    return perm


def get_clusters(instance):
    """Get the clusters for this instance"""
    perm = {s: [] for s in instance.clusters}
    for s in instance.clusters:
        for j in instance.J:
            for i in instance.I:
                if instance.z[i, j, s].value > 0.5:
                    perm[s].append(i)
                    break
    return Clustering(perm)


def is_feasible(model, constr_tol=1e-8, var_tol=1e-8):
    """
    Checks to see if the algebraic model is feasible in its current state.
    Checks variable bounds and active constraints. Not for use with
    untransformed GDP models.
    """
    for constr in model.component_data_objects(
        ctype=pyenv.Constraint, active=True, descend_into=True
    ):
        # Check constraint lower bound
        if constr.lower is not None and (
            pyenv.value(constr.lower) - pyenv.value(constr.body) >= constr_tol
        ):
            print(
                constr,
                "Lower bound not satisfied:",
                pyenv.value(constr.lower),
                ">",
                pyenv.value(constr.body),
            )
        # check constraint upper bound
        if constr.upper is not None and (
            pyenv.value(constr.body) - pyenv.value(constr.upper) >= constr_tol
        ):
            print(
                constr,
                "Upper bound not satisfied:",
                pyenv.value(constr.upper),
                "<",
                pyenv.value(constr.body),
            )
    for var in model.component_data_objects(ctype=pyenv.Var, descend_into=True):
        # Check variable lower bound
        if var.has_lb() and pyenv.value(var.lb) - pyenv.value(var) >= var_tol:
            print(
                var,
                "Lower bound not satisfied:",
                pyenv.value(var.lb),
                ">",
                pyenv.value(var),
            )
        # Check variable upper bound
        if var.has_ub() and pyenv.value(var) - pyenv.value(var.ub) >= var_tol:
            print(
                var,
                "Upper bound not satisfied:",
                pyenv.value(var.ub),
                "<",
                pyenv.value(var),
            )


def reorder_clusters(clusters):
    """Reorder the clusters"""
    new_clusters = {}
    for j, (_, cluster) in enumerate(sorted(clusters.items(), key=lambda x: min(x[1]))):
        new_clusters[j + 1] = cluster
    return Clustering(new_clusters)


def clusters_as_dictionary(model, clusters):
    """
    Rewrite clusters given in format {cluster_no -> ordered_pts_in_cluster}
    into two dictionaries u and z which represent the binary variables of the
    model.
    """
    u = {}
    z = {}

    for i in model.I0:
        for s in model.clusters:
            if i in clusters[s]:
                u[i, s] = 1
            else:
                u[i, s] = 0

    for s in model.clusters:
        z[0, 0, s] = 1
        for j in model.J:
            z[0, j, s] = 0
        for i in model.I:
            z[i, 0, s] = 0

    for i in model.I:
        for j in model.J:
            for s in model.clusters:
                if len(clusters[s]) >= j and clusters[s][j - 1] == i:
                    z[i, j, s] = 1
                else:
                    z[i, j, s] = 0
    return u, z


def add_local_branching_heuristic(model, u, d=2, z=None):
    """Add a local branching heuristic"""
    expr = sum(model.u[i, s] for ((i, s), val) in u.items() if val == 0) + sum(
        1 - model.u[i, s] for ((i, s), val) in u.items() if val == 1
    )
    if z is not None:
        expr += sum(
            model.z[i, j, s] for ((i, j, s), val) in z.items() if val == 0
        ) + sum(1 - model.z[i, j, s] for ((i, j, s), val) in z.items() if val == 1)

    model.local_branch_constraint = pyenv.Constraint(expr=expr <= d)


def cost_function(x, energies, epsilon=0.01):
    """Cost function of difference in energies"""
    if min(abs(x - energy) for energy in energies) > epsilon:
        return 1
    return 0


def compute_different_energy_spectra(
    events, tracks, alpha=0.4, S=2, link=False, distance="great_circle"
):
    """
    Compute the following energy spectra:
        full_energies: the energy for each event (no clustering)
        subset_sum_energies: the energy of each cluster using subset sum
            clusters (variable tracks)
    """
    full_energies = []
    subset_sum_energies = []
    track_and_cluster_energies = []
    cluster_energies = []
    for t, event in zip(tracks, events):
        if len(event.hit_points) == 0:
            continue
        energy = sum(p.e for p in event.hit_points)
        full_energies.append(energy)

        if link:
            clustered_events = event.cluster_linkage(alpha, S, distance=distance)
        else:
            clustered_events = event.cluster(alpha)
        clustered_energies = event.energy_sums(clustered_events)
        cluster_energies.extend(clustered_energies.values())

        if t is not None:
            if isinstance(t, Clustering):
                track = t
                energies = [
                    sum(event.points[i].e for i in cluster)
                    for cluster in track.values()
                    if len(cluster) > 0
                ]
                cost = sum(cost_function(e, [1.173, 1.333]) for e in energies)
            else:
                (track, cost) = t
            subset_energies = [
                sum(event.points[i].e for i in cluster)
                for cluster in track.values()
                if len(cluster) > 0
            ]
            subset_sum_energies.extend(subset_energies)

            if cost == 2:
                track_and_cluster_energies.extend(clustered_energies.values())
            else:
                track_and_cluster_energies.extend(subset_energies)

    return (
        full_energies,
        subset_sum_energies,
        track_and_cluster_energies,
        cluster_energies,
    )


def clusters_match(clustering1, clustering2):
    """Do the two clusterings match"""
    # Remove all null clusters which have no elements
    c1 = {k: cluster for (k, cluster) in clustering1.items() if len(cluster) > 0}
    c2 = {k: cluster for (k, cluster) in clustering2.items() if len(cluster) > 0}
    if len(c1) != len(c2):
        return False
    for cluster in c1.values():
        if all([sorted(cluster) != sorted(cluster2) for cluster2 in c2.values()]):
            return False
    return True


def get_track_energies(events, tracks):
    """
    Return a list of the energies given by the specified tracks
    """
    energies = []
    for i, track in enumerate(tracks):
        if track is None:
            continue
        energies.extend(events[i].energy_sums(track).values())
    return energies


def mismatch_count(
    approx_clusters: dict, true_clusters: dict, return_cluster_indices: bool = False
):
    """
    For each cluster in approx_clusters, find the cluster in true_clusters
    which is closest in terms of size of symmetric difference.

    Return dictionary mapping cluster_idx to the distance to best true cluster
    and dictionary mapping cluster_idx to the best true cluster

    Args:
        approx_clusters (dict): Clusters created by a clustering method
        true_clusters (dict): Ground truth clustering
    """
    mismatches = {}
    matched_clusters = {}
    matched_cluster_indices = {}
    for i, cluster in approx_clusters.items():
        best_match_cost = np.inf
        for true_index, true_cluster in true_clusters.items():
            # Size of the symmetric difference
            cost = len(set(true_cluster) ^ set(cluster))
            if cost < best_match_cost:
                best_match_cost = cost
                best_cluster = true_cluster
                best_cluster_index = true_index
        mismatches[i] = best_match_cost
        matched_clusters[i] = best_cluster
        matched_cluster_indices[i] = best_cluster_index
    if return_cluster_indices:
        return mismatches, matched_cluster_indices
    return mismatches, matched_clusters


def compute_reclustered_energies(events, tracks, ray_tracks, observed_energies):
    """
    Categorize clusters by the type of event they emerged from, either an escape
    event, or a full absorption event.

    Returns a dictionary mapping type of event to a list of energies that
    the event had been broken up into.

    Args:
        events (list): The list of Events
        tracks (dict): The clustering given by a tracking algorithm,
            {cluster_idx -> cluster}
        ray_tracks (dict): The ground truth clustering
            {cluster_idx -> cluster}
        observed_energies (list): The list of observed energies
    """
    reclustered_energies = {energy: [] for energy in observed_energies}
    reclustered_energies["Other"] = []
    for event, track, ray_track in zip(events, tracks, ray_tracks):
        if track is None:
            continue
        _, matched_clusters = mismatch_count(track, ray_track)
        for i, cluster in track.items():
            if len(cluster) == 0:
                continue
            matched_cluster = matched_clusters[i]
            energy = sum(event.points[i].e for i in cluster)
            matched_energy = sum(event.points[i].e for i in matched_cluster)
            for observed_energy in observed_energies:
                if np.isclose(matched_energy, observed_energy):
                    reclustered_energies[observed_energy].append(energy)
                    break
            else:
                reclustered_energies["Other"].append(energy)
    return reclustered_energies


def compute_mismatch_counts(events, tracks, ray_tracks):
    """
    Categorize clusters by how many elements they are off from a correct cluster.

    Returns dictionary mapping mismatch count to list of energies from clusters
    that had that many mismatches.
    """
    mismatch_energies = {}
    for event, track, ray_track in zip(events, tracks, ray_tracks):
        if track is None:
            continue
        mismatch_counts, _ = mismatch_count(track, ray_track)
        for i, cluster in track.items():
            if len(cluster) == 0:
                continue
            energy = sum(event.points[i].e for i in cluster)
            error_count = mismatch_counts[i]
            if error_count in mismatch_energies:
                mismatch_energies[error_count].append(energy)
            else:
                mismatch_energies[error_count] = [energy]
    return mismatch_energies


def find_misclustered_indices(tracks, true_tracks):
    """
    Find the indices for which the clustering doesn't match the true clustering.
    """
    bad_indices = []
    for i, (track, true_track) in enumerate(zip(tracks, true_tracks)):
        if track is None or not clusters_match(track, true_track):
            bad_indices.append(i)
    return bad_indices


def compute_subset_and_cluster_tracks(
    events, subset_tracks, observed_energies, alpha=0.4, distance="great_circle"
):
    """Compute subset and cluster tracks

    Args:
        events (list(Event)): The gamma ray events
        subset_tracks (list(Clusters)): The clusters computed using the SubSum
            approach
        observed_energies (list): The expected energies (e.g., [1.1732266,
            1.332498] for Co60)
        alpha (float): Clustering angle in radians
    """
    subset_and_cluster_tracks = []
    for event, track in zip(events, subset_tracks):
        # Compute clusters using distance
        clusters = event.cluster_linkage(alpha, 2, distance=distance)
        # clusters = event.cluster(alpha) # Compute clusters using GRETINA method
        if track is None:
            subset_and_cluster_tracks.append(clusters)
            continue

        # Accept the SubSum clusters if the energies match the expected values,
        # otherwise accept the distance clustering
        energies = list(
            event.energy_sums(track).values()
        )  # Compute energies for tracks
        if sum(cost_function(e, observed_energies) for e in energies) == len(energies):
            # If the energy sum doesn't match the observed_energies (here the 2
            # indicates that energy does not match for either expected gamma
            # ray)...
            subset_and_cluster_tracks.append(clusters)
            # ...use the clustering using distance
        else:
            subset_and_cluster_tracks.append(track)
            # ...otherwise accept the subset sum clusters (energies match)
    return subset_and_cluster_tracks


def combine_clusterings(events1, events2, track1, track2):
    """
    This function combines the clusterings for one list of Events with the
    clusterings for the second list of Events. It is assumed that the two lists
    of events were created by splitting events into disjoint sets of interactions.
    Events of the same index in events1 and events2 should have come from the
    same original Event.
    """
    combined_tracks = []
    for e1, e2, t1, t2 in zip(events1, events2, track1, track2):
        combined_track = {}
        for e, t in [(e1, t1), (e2, t2)]:
            for _, cluster in t.items():
                if len(cluster) == 0:
                    continue
                combined_track[max(combined_track, default=0) + 1] = [
                    e.points[j].number for j in cluster
                ]
        combined_tracks.append(Clustering(combined_track))
    return combined_tracks


def remove_peak_events(events, tracks, observed_energies, epsilon=0.01):
    """
    This function runs through all the events and removes any cluster which
    sums up to an energy within a tolerance, epsilon, of an observed
    energy.

    This splits each of the events into the an Event with the peak gamma rays
    clusters and an Event with the remaining clusters.
    It also returns a list of dictionaries containing the peak clusters
    for each event.

    Args:
        events (list[event]): list of events
        tracks (list[clusters]): list of dictionary of clusters
        observed_energies (list[float]): observed (or expected) energies in the peaks
        epsilon (float): tolerance defining the peak half-width

    Returns:
        peak_events (list[event]): list of events containing peak clusters
        bg_events (list[event]): list of events with peak clusters removed
        peak_clusters (list[clusters]): list of clusters with peak energies (if
            the event does not have clusters with peak energies, an empty clustering
            will be included)
    """
    peak_events = []
    bg_events = []
    peak_clusters = []
    for event, track in zip(events, tracks):
        energy_sums = event.energy_sums(track)
        peak_indices = []
        peak_cluster = {}
        current_index = 1
        for i, energy in energy_sums.items():
            for obs_energy in observed_energies:
                if abs(energy - obs_energy) < epsilon:
                    peak_indices.extend(track[i])
                    peak_cluster[i] = [j + current_index for j in range(len(track[i]))]
                    current_index += len(track[i])
                    break
        background_indices = [
            i for i in range(1, len(event.points)) if i not in peak_indices
        ]
        peak_event = event.subset(peak_indices)
        peak_events.append(peak_event)
        bg_event = event.subset(background_indices)
        bg_events.append(bg_event)
        peak_clusters.append(Clustering(peak_cluster))
    return peak_events, bg_events, peak_clusters


def invert_clusters(clusters: dict):
    """Invert the cluster from {cluster index: items} to {items: cluster index}"""
    p_to_cluster_idx = {}
    for c, vals in clusters.items():
        for val in vals:
            p_to_cluster_idx[val] = c
    return p_to_cluster_idx


def labels_to_clusters(labels: dict):
    """
    Invert labels from {items: cluster index} to {cluster index: items}
    """
    clusters = {}
    for i, c in labels.items():
        if c not in clusters.keys():  # pylint: disable=consider-iterating-dictionary
            clusters[c] = [i]
        else:
            clusters[c].append(i)
    return clusters


def compute_purity(clusters, true_clusters):
    """Compute the purity of the clusters"""
    purity = 0
    for c in clusters.keys():
        closest_cluster = max(
            true_clusters, key=lambda x: len(set(true_clusters[x]) & set(clusters[c]))
        )
        purity += len(set(true_clusters[closest_cluster]) & set(clusters[c]))
    purity /= max(max(c) for c in clusters.values())
    return purity


def cluster_summary(clusters, true_clusters):
    """
    Return true positives, true negatives, false positives, false negatives
    for a clustering compared to the true clustering.

    This is computed for each pair of elements.
    """
    inv_clusters = invert_clusters(clusters)
    inv_true_clusters = invert_clusters(true_clusters)

    tp = tn = fp = fn = 0

    points = list(inv_true_clusters.keys())
    for p1, p2 in combinations(points, 2):
        if inv_true_clusters[p1] == inv_true_clusters[p2]:
            if inv_clusters[p1] == inv_clusters[p2]:
                tp += 1
            else:
                fn += 1
        else:
            if inv_clusters[p1] == inv_clusters[p2]:
                fp += 1
            else:
                tn += 1
    return tp, tn, fp, fn


def compute_precision(clusters, true_clusters):
    """Compute the precision"""
    tp, _, fp, _ = cluster_summary(clusters, true_clusters)
    if tp + fp == 0:
        return 1

    return tp / (tp + fp)


def compute_recall(clusters, true_clusters):
    """Compute the recall"""
    tp, _, _, fn = cluster_summary(clusters, true_clusters)
    if tp + fn == 0:
        return 1

    return tp / (tp + fn)


def compute_rand_index(clusters, true_clusters):
    """Compute the rand index"""
    tp, tn, fp, fn = cluster_summary(clusters, true_clusters)
    if tp + tn + fp + fn == 0:
        return 1

    return (tp + tn) / (tp + tn + fp + fn)


def compute_jaccard_index(clusters, true_clusters):
    """Compute the Jaccard index"""
    tp, _, fp, fn = cluster_summary(clusters, true_clusters)
    if tp + fp + fn == 0:
        return 1

    return tp / (tp + fp + fn)


def count_fully_recovered_clusters(clusters, true_clusters):
    """Count the fully recovered clusters"""
    mismatch_counts, _ = mismatch_count(true_clusters, clusters)
    return len([c for c, val in mismatch_counts.items() if val == 0])


def compute_cluster_statistics(tracks, true_tracks):
    """Compute cluster statistics"""
    precisions = []
    recalls = []
    rands = []
    jaccards = []
    recovered_clusters = []
    for clusters, true_clusters in zip(tracks, true_tracks):
        precisions.append(compute_precision(clusters, true_clusters))
        recalls.append(compute_recall(clusters, true_clusters))
        rands.append(compute_rand_index(clusters, true_clusters))
        jaccards.append(compute_jaccard_index(clusters, true_clusters))
        recovered_clusters.append(
            count_fully_recovered_clusters(clusters, true_clusters)
        )
    return precisions, recalls, rands, jaccards, recovered_clusters


def get_event_ray_subset(events, tracks, ray_nums):
    """
    Return subset of events and tracks which is given by selecting out the
    gamma rays of certain indices.
    """
    new_events = []
    new_tracks = []
    for e, t in zip(events, tracks):
        indices = sum([t.get(num, []) for num in ray_nums], start=[])
        new_t = {
            num: [indices.index(p) + 1 for p in t[num]] for num in ray_nums if num in t
        }
        new_e = e.subset(indices, reset_idxs=True)  # TODO - event class restructured
        new_events.append(new_e)
        new_tracks.append(new_t)
    return new_events, new_tracks


def in_peak(energy: float, observed_energies: List[float], tolerance:float=0.005):
    """
    Return True if energy is within tolerance of one of the observed_energies

    Args:
        energy (float): The energy to be checked if they are in the supplied peaks
        observed_energies (list[float]): The energies we are checking against (peaks)
        tolerance (float): The tolerance for the difference between energy and peak energy
    """
    return np.any(np.abs(energy - observed_energies) < tolerance)


def get_all_clusters(linkage, n_points):
    """
    Given a linkage array from a hierarchical clustering algorithm, return
    the list of all clusters produced.
    """
    clusters = {i: [i + 1] for i in range(n_points)}
    for i in range(n_points - 1):
        c1_idx, c2_idx = linkage[i, [0, 1]]
        clusters[n_points + i] = clusters[c1_idx] + clusters[c2_idx]
    return clusters


def angular_distance(x1, x2):
    """
    Angle between two points in R3.
    """
    return abs(np.arccos(np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))))


# def combine_events_clusters(events1: list, events2: list,
#                             tracks1: list[dict], tracks2: list[dict]):
#     """
#     This function combines the clusterings for one list of Events with the
#     clusterings for the second list of Events. It is assumed that the two lists
#     of events were created by splitting events into disjoint sets of interactions.
#     Events of the same index in events1 and events2 should have come from the
#     same original Event.

#     Does not recreate the original event if event1 and event2 were created by
#     splitting a single list of events.

#     Args:
#         events1 (list(Event)): one list of events
#         events2 (list(Event)): the other list of events
#         track1 (list(clusters)): list of clusters corresponding to events1
#         track2 (list(clusters)): list of clusters corresponding to events2
#     Returns:
#         combined_events (list(Event)): combined events list
#         combined_clusters (list(clusters)): combined clusters list
#     """
#     from gamma_ray import Event
#     combined_tracks = []
#     combined_events = []
#     for e1, e2, t1, t2 in zip(events1, events2, tracks1, tracks2):
#         offset = len(e1.hit_points)
#         combined_track = {}
#         combined_hit_points = []
#         track_index = 1
#         for point in e1.hit_points:
#             combined_hit_points.append(point)
#         for track in t1:
#             combined_track[track_index] = t1[track]
#             track_index += 1
#         for point in e2.hit_points:
#             combined_hit_points.append(point)
#         for track in t2:
#             combined_track[track_index] = [index + offset for index in t2[track]]
#             track_index += 1
#         combined_events.append(Event(len(combined_events), combined_hit_points))
#         combined_tracks.append(combined_track)

#     return combined_events, combined_tracks


def combine_events(
    event1: Event,
    event2: Event,
    tracks1: dict = None,
    tracks2: dict = None,
    return_escapes_label: bool = False,
):
    """Combine two events"""
    total_event = Event(event1.id, event1.hit_points + event2.hit_points)
    escapes_label = {}
    if tracks1 is not None and tracks2 is not None:
        total_tracks = {}
        i = 1
        offset = 0
        for track in tracks1.values():
            total_tracks[i] = track
            escapes_label[i] = 0
            i += 1
            offset += len(track)
        for track in tracks2.values():
            total_tracks[i] = [index + offset for index in track]
            escapes_label[i] = 1
            i += 1
        if return_escapes_label:
            return total_event, total_tracks, escapes_label
        return total_event, total_tracks
    return total_event


def fix_order(event, al_event, al_tracks: dict) -> dict:
    """
    Convert indices related to the altered event in the altered tracks back to
    the indices related to the original event.

    This is done by energy matching (not always accurate)
    TODO - change matching strategy

    Does not change cluster indices, just the indices to event points in the clusters.
    """
    index_map = {}
    for true, p in enumerate(event.points):
        for altered, ap in enumerate(al_event.points):
            if p.e == ap.e and p.e != 0:
                index_map[altered] = true
    fixed_tracks = {}
    for index, track in al_tracks.items():
        fixed_tracks[index] = [index_map[altered] for altered in track]
    return fixed_tracks


def cluster_matrix(
    clusters: Dict[int, list],
    cluster_validity: bool = True,
    order_validity: bool = True,
) -> np.ndarray:
    """
    Return the connectivity matrix for the provided clusters
    """
    max_index = 0
    for cluster in clusters.values():
        max_index = max(*cluster, max_index)
    y = np.zeros((max_index + 1, max_index + 1), dtype=int)
    for _, cluster in clusters.items():
        if cluster_validity:
            for i in [0] + list(cluster):
                for j in cluster:
                    y[i, j] += 1
        if order_validity:
            for i, j in zip([0] + list(cluster), cluster):
                y[i, j] += 1
    return y


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


def remove_zero_energy(events, list_of_clusters=None, **kwargs):
    """
    Remove interactions from events with zero energy
    """
    events_copy = []
    clusters_copy = []
    for j, event in enumerate(events):
        removal_indices = []
        for i, point in enumerate(event.hit_points):
            if point.e <= 0:
                removal_indices.append(i + 1)
        if list_of_clusters is not None:
            ev, clu = remove_interactions(
                event, removal_indices, list_of_clusters[j], **kwargs
            )
            clusters_copy.append(clu)
        else:
            ev = remove_interactions(event, removal_indices, **kwargs)
        events_copy.append(ev)
    if list_of_clusters is not None:
        return events_copy, clusters_copy
    return events_copy


def remove_zero_energy_interactions(event, clusters=None, energy_threshold: float = 0):
    """
    Remove zero energies for a single event
    """
    removal_indices = []
    for i, point in enumerate(event.hit_points):
        if point.e <= energy_threshold:
            removal_indices.append(i + 1)
    if clusters is not None:
        return remove_interactions(event, removal_indices, clusters)
    return remove_interactions(event, removal_indices)


def join_validity(
    ground_truth_clusters: Dict[int, Iterable[int]],
    cluster1: Iterable[int],
    cluster2: Iterable[int],
):
    """
    Should proposed clusters 1 and 2 be joined into the same cluster?

    Returns True if all indices from both cluster1 and cluster2 are in the same
    cluster in ground_truth_clusters
    """
    for key in ground_truth_clusters.keys():
        if all(i in ground_truth_clusters[key] for i in cluster1) and all(
            j in ground_truth_clusters[key] for j in cluster2
        ):
            return True
    return False


def end_validity(
    ground_truth_clusters: Dict[int, Iterable[int]],
    cluster1: Iterable[int],
    cluster2: Iterable[int],
):
    """
    Should proposed clusters 1 and 2 be joined end to end in the same cluster?

    Returns True if cluster1 and cluster2 are both part of the same ground truth
    clustering and should be joined in order [cluster1] + [cluster2]
    """
    if join_validity(ground_truth_clusters, cluster1, cluster2):
        for key in ground_truth_clusters.keys():
            for i, index in enumerate(ground_truth_clusters[key][:-1]):
                if cluster1[-1] == index:
                    if cluster2[0] == ground_truth_clusters[key][i + 1]:
                        return True
                    else:
                        return False
    return False


def valid_transition(ground_truth_clusters, i, j):
    """
    Is the transition from i -> j valid?
    """
    if i == 0:
        for key in ground_truth_clusters.keys():
            if ground_truth_clusters[key][0] == j:
                return True
    for key, cluster in ground_truth_clusters.items():
        if i in cluster and j in cluster:
            for pos, index in enumerate(cluster[:-1]):
                if index == i:
                    if cluster[pos + 1] == j:
                        return True
                    else:
                        return False
        if (i in cluster and j not in cluster) or (j in cluster and i not in cluster):
            return False
    return False


def same_cluster(ground_truth_clusters, i, j):
    """
    Are the interactions i and j in the same cluster?
    """
    if i == 0 or j == 0:
        return True
    for _, cluster in ground_truth_clusters.items():
        if i in cluster:
            if j in cluster:
                return True
            return False
        if j in cluster:
            if i in cluster:
                return True
            return False
    return False


def reindex_clusters(clusters: dict):
    """
    Change the indices of clusters by minimum point index
    """
    new_clusters = {}
    keys = list(clusters.keys())
    new_key = 1
    while len(keys) > 0:
        min_index = 1e90
        selected_key = 0
        for key in keys:
            if clusters[key][0] < min_index:
                selected_key = key
                min_index = min(clusters[key])
        new_clusters[new_key] = clusters[selected_key]
        new_key += 1
        keys.remove(selected_key)
    return new_clusters


def fraction_of_true_clusters_captured_dict(
    clusters_true: dict[int, list], clusters_pred: dict[int, list], debug: bool = False
) -> tuple[dict, dict, dict]:
    """Evaluate the predicted clusters compared to the true clusters

    Arguments are the true clusters dictionary and the predicted clusters dictionary

    Returns dictionaries indicating if the true clusters are:
    - correctly captured in the predicted clusters
    - split up in the predicted clusters
    - joined together in the predicted clusters
    - ordered correctly
    """
    labels_pred = invert_clusters(
        clusters_pred
    )  # The predicted cluster ids for each point
    labels_true = invert_clusters(clusters_true)  # The cluster ids for each point
    correct = (
        split_up
    ) = joined = 0  # A cluster can be both split up and joined, but not correct
    correct = {}  # Is the true cluster captured as a single cluster?
    ordered = (
        {}
    )  # Is the true cluster captured as a single cluster and ordered correctly?
    split_up = {}  # Is the true cluster split up across multiple predicted clusters?
    joined = (
        {}
    )  # Is the true cluster joined together with other true clusters in the predicted clusters?

    # What are the true cluster sources for each predicted cluster?
    pred_cluster_sources = {k: set() for k in clusters_pred.keys()}
    for cid, pred_cluster in clusters_pred.items():
        for index in pred_cluster:
            pred_cluster_sources[cid].add(labels_true[index])
    if debug:
        print("Sources", pred_cluster_sources)

    # What are the predicted cluster targets for each true cluster?
    true_cluster_targets = {k: set() for k in clusters_true.keys()}
    for cid, true_cluster in clusters_true.items():
        for index in true_cluster:
            true_cluster_targets[cid].add(labels_pred[index])
    if debug:
        print("Targets", true_cluster_targets)

    # Split up means that indices from the same true cluster appear in multiple predicted clusters
    # i.e., multiple targets in true_cluster_targets
    split_up = {}
    for k, sources in true_cluster_targets.items():
        # The true cluster has multiple targets in the predicted clusters
        split_up[k] = len(sources) > 1

    # Joined means that indices from different true clusters appear in the same predicted cluster
    # i.e., multiple sources in pred_cluster_sources
    joined = {k: False for k in clusters_true.keys()}
    for k, sources in pred_cluster_sources.items():
        if len(sources) > 1:
            for source in sources:
                joined[source] = True

    if debug:
        print("Split ", split_up)
        print("Joined", joined)
    # Complete means not split up or joined
    correct = {k: ((not split_up[k]) and (not joined[k])) for k in joined.keys()}
    # Ordered means correct and the order of interactions is also correct
    for k in clusters_true.keys():
        if correct[k]:
            ordered[k] = all(
                clusters_pred[labels_pred[clusters_true[k][i]]][i]
                == clusters_true[k][i]
                for i in range(len(clusters_true[k]))
            )
        else:
            ordered[k] = False
    return correct, split_up, joined, ordered
