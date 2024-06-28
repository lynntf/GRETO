"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Event class
"""

from __future__ import annotations

from functools import cached_property, lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage  # fcluster,
from scipy.spatial.distance import squareform  # pdist,

import greto.geometry as geo
import greto.physics as phys
from greto import default_config
from greto.asym_heir_clustering import asym_hier_linkage
from greto.coincidence_class import Coincidence
from greto.detector_config_class import DetectorConfig
from greto.interaction_class import Interaction
from greto.utils import perm_to_transition, reverse_cumsum

# TODO - change from eres constant to energy uncertainty


class Event:
    """
    # Class which encodes a set of interactions which were observed
    as the result of a sequence of &gamma;-rays emitted from a source.

    ## Attributes:
        - id: An event id
        - points: A list of &gamma;-ray Interactions in time coincidence with
          one another. The zeroth index of this list is always reserved for the
          origin, and is a shared interaction for each &gamma;-ray.
        - hit_points: The list of Interactions excluding the origin
        - ground_truth (Clustering): If the ground truth clustering and
          ordering is known, it can be passed in.
        - flat: If the event has been flattened, the third dimension has been
          removed for 2D plotting
    """

    def __init__(
        self,
        event_id: int,
        points: List[Interaction],
        ground_truth: dict = None,
        flat: bool = False,
        detector: DetectorConfig = default_config,
    ):
        # self.id = str(event_id)
        self.id = event_id
        if isinstance(detector, str):
            detector = DetectorConfig(detector)
        if len(points) == 0 or not np.all(points[0].x == detector.origin.x):
            # Insert an interaction at the origin
            self.points = [detector.origin] + points
        else:
            # If already have an origin Interaction, keep points list
            self.points = points
        self.energy = sum(point.e for point in self.points)
        self.ground_truth = ground_truth
        self.linkage = None
        self.flat = flat
        self.detector_config = detector

    def __repr__(self):
        return f"Event(event_id={self.id}, points={self.points})"

    def __str__(self) -> str:
        s = f"<Event {self.id}:"
        for i, point in enumerate(self.points):
            s += f"\n{i:2d}: {point}"
        s += ">"
        return s

    def __len__(self) -> int:
        return len(self.hit_points)

    @property
    def origin(self) -> Interaction:
        """Origin of the detector array"""
        return self.points[0]

    @property
    def hit_points(self) -> List[Interaction]:
        """The interaction points which are not the origin."""
        return self.points[1:]

    @cached_property
    def point_matrix(self) -> np.ndarray:
        """Matrix of interaction point coordinates including the origin"""
        return np.vstack([p.x for p in self.points])

    @cached_property
    def radii(self) -> np.ndarray:
        """Interaction point radius"""
        return geo.radii(self.point_matrix)

    @property
    def crystal_hit_matrix(self) -> np.ndarray:
        """Matrix of interaction point coordinates in the crystal frame
        (excludes origin)."""
        return np.vstack([p.crystal_x for p in self.hit_points])

    @property
    def spherical_point_matrix(self) -> np.ndarray:
        """Matrix of interaction point coordinates in spherical including the origin"""
        return geo.cartesian_to_spherical(self.point_matrix)

    @property
    def hit_point_matrix(self) -> np.ndarray:
        """Matrix of interaction points in the detector"""
        return self.point_matrix[1:]

    @cached_property
    def energy_matrix(self) -> np.ndarray:
        """Matrix of interaction point energies"""
        return np.fromiter(
            (p.e for p in self.points), dtype=float, count=len(self.points)
        )

    def energy_sum(self, perm: Iterable[int]) -> float:
        """Sum of energies for points in perm"""
        return np.sum(self.energy_matrix[list(perm)])

    def energy_sums(self, clusters: Dict[int, Iterable[int]]) -> float:
        """Sum of energies for points in a dict of clusters"""
        return {s: self.energy_sum(perm) for s, perm in clusters.items()}

    @cached_property
    def position_uncertainty(self) -> np.ndarray:
        """Position uncertainty of interactions"""
        return self.detector_config.position_error(self.energy_matrix)

    @property
    def energy_uncertainty(self) -> np.ndarray:
        """Energy uncertainty of interactions using AGATA model"""
        return np.sqrt(1 + 3.7 * self.energy_matrix) / 2.3548 / 1000

    @property
    def hit_energy_matrix(self) -> np.ndarray:
        """Matrix of interaction point energies in the detector"""
        return self.energy_matrix[1:]

    @cached_property
    def data_matrix(self) -> np.ndarray:
        """Matrix of interaction points"""
        return np.concatenate(
            (self.point_matrix, self.energy_matrix[:, np.newaxis]), axis=1
        )

    @property
    def distance(self) -> np.ndarray:
        """Distance between two points"""
        # return squareform(geo.pairwise_distance(self.point_matrix))
        _calculator.set_event(self)
        return _calculator.distance((self.id, tuple(self.points[1].x)))

    def distance_perm(
        self, permutation: Iterable[int], start_point: int = 0
    ) -> np.ndarray:
        """Distances between points in a permutation"""
        if start_point is not None:
            full_perm = tuple([start_point] + list(permutation))
        else:
            full_perm = tuple(permutation)
        return self.distance[perm_to_transition(full_perm, D=2)]

    @property
    def angle_distance(self) -> np.ndarray:
        """
        Angular distance between two points
        TODO - can use cos_act property for this to avoid possible extra computation
        """
        # return squareform(
        #     np.arccos(1.0 - pdist(self.hit_point_matrix, metric="cosine"))
        # )
        _calculator.set_event(self)
        return _calculator.angle_distance((self.id, tuple(self.points[1].x)))

    @property
    def ge_distance(self) -> np.ndarray:
        """Distance between two points"""
        # return squareform(
        #     geo.ge_distance(self.point_matrix, d12_euc=squareform(self.distance))
        # )
        _calculator.set_event(self)
        return _calculator.ge_distance((self.id, tuple(self.points[1].x)))

    def ge_distance_perm(
        self, permutation: Iterable[int], start_point: int = 0
    ) -> np.ndarray:
        """Distances between points in a permutation"""
        if start_point is not None:
            full_perm = tuple([start_point] + list(permutation))
        else:
            full_perm = tuple(permutation)
        return self.ge_distance[perm_to_transition(full_perm, D=2)]

    def linkage_array(
        self,
        distance: str = "great_circle",
        method: str = "single",
        time_gap: int = 40,
        center_cross_penalty: float = 0.0,
        center_cross_threshold: float = 1e-3,
        center_cross_factor: float = 0.0,
        **kwargs,
    ) -> np.ndarray:
        """Clustering linkage"""
        if len(self.hit_points) <= 1:
            return None
        # time_gap_accept = (pdist(np.expand_dims(np.array([p.ts for p in self.hit_points]), axis=1)) < time_gap)
        time_gap_accept = (geo.njit_pdist(np.expand_dims(np.array([p.ts for p in self.hit_points]), axis=1)) < time_gap)
        if distance.lower() == "euclidean":
            # distances = squareform(self.distance[1:, 1:])
            distances = geo.njit_squareform_matrix(self.distance[1:, 1:])
        elif distance.lower() in ["great_circle", "cosine"]:
            # distances = np.arccos(1 - pdist(self.hit_point_matrix, metric="cosine"))
            distances = np.arccos(1 - geo.njit_cosine_pdist(self.hit_point_matrix))
        elif distance.lower() == "germanium":
            # distances = squareform(self.ge_distance[1:, 1:])
            distances = geo.njit_squareform_matrix(self.ge_distance[1:, 1:])
            if center_cross_penalty > 0.0 or center_cross_factor > 0.0:
                center_cross_factor = np.nan_to_num(center_cross_factor)
                center_cross_penalty = np.nan_to_num(center_cross_penalty)
                euc_distances = geo.njit_squareform_matrix(self.distance[1:, 1:])
                if center_cross_factor > 0.0:
                    distances += (
                        np.maximum(euc_distances - distances, 0) * center_cross_factor
                    )
                if center_cross_penalty > 0.0:
                    distances[
                        euc_distances - distances > center_cross_threshold
                    ] += center_cross_penalty
        else:
            raise NotImplementedError

        distances[~time_gap_accept] += np.nan_to_num(np.inf)

        if method.startswith("dir") or method.startswith("asym"):
            # Z = asym_hier_linkage(squareform(distances), **kwargs)
            Z = asym_hier_linkage(geo.njit_squareform_vector(distances), **kwargs)
        else:
            try:
                Z = linkage(distances, method=method)
            except ValueError as e:
                print(f"problem with distances {squareform(distances)}")
                print(f"Event: {self}")
                raise e

        self.linkage = Z
        return Z

    # %% Convenience functions
    def cluster_linkage(
        self,
        alpha: float = np.deg2rad(10.0),
        alpha_degrees: float = None,
        max_clusters: int = 30,
        **linkage_kwargs: Any,
    ) -> Dict[int, int]:
        """Cluster using a linkage"""
        from greto.cluster_tools import cluster_linkage

        return cluster_linkage(
            self,
            alpha=alpha,
            alpha_degrees=alpha_degrees,
            max_clusters=max_clusters,
            **linkage_kwargs,
        )

    def pack_interactions(
        self,
        packing_distance: float = 0.6,
        clusters: dict = None,
        keep_duplicates: bool = False,
    ):
        """Pack interactions"""
        from greto.cluster_tools import pack_interactions

        return pack_interactions(
            self,
            packing_distance=packing_distance,
            clusters=clusters,
            keep_duplicates=keep_duplicates,
        )

    # %% Cached geometric cosine and error
    @property
    def cos_act(self) -> np.ndarray:
        """Cosine between interaction points"""
        # return geo.cosine_ijk(self.point_matrix)
        _calculator.set_event(self)
        return _calculator.cos_act((self.id, tuple(self.points[1].x)))

    @property
    def cos_err(self) -> np.ndarray:
        """Error in cosine due to position uncertainty"""
        # err = geo.err_cos_vec_precalc(
        #     self.distance, self.cos_act, self.position_uncertainty
        # )
        # # TODO - should the center point be made more accurate here or in position_uncertainty?
        # # err[0,:,:] /= 2 # Center point is more accurate
        # return err
        _calculator.set_event(self)
        return _calculator.cos_err((self.id, tuple(self.points[1].x)))

    @property
    def theta_err(self) -> np.ndarray:
        """Error in cosine due to position uncertainty"""
        # err = geo.err_theta_vec_precalc(
        #     self.distance, self.cos_act, self.position_uncertainty
        # )
        # # TODO - should the center point be made more accurate here or in position_uncertainty?
        # # err[0,:,:] /= 2 # Center point is more accurate as in AFT
        # return err
        _calculator.set_event(self)
        return _calculator.theta_err((self.id, tuple(self.points[1].x)))

    # %% Permuted geometric cosine and error
    def cos_act_perm(
        self, permutation: Iterable[int], start_point: int = 0
    ) -> np.ndarray:
        """The cosines of angles from permutation"""
        if start_point is not None:
            full_perm = tuple([start_point] + list(permutation))
        else:
            full_perm = tuple(permutation)
        try:
            return self.cos_act[perm_to_transition(full_perm, D=3)]
        except IndexError as ex:
            print(self)
            print(self.cos_act)
            print(full_perm)
            raise ex

    def cos_act_err_perm(
        self, permutation: Iterable[int], start_point: int = 0
    ) -> np.ndarray:
        """Standard error of cosine"""
        if start_point is not None:
            full_perm = tuple([start_point] + list(permutation))
        else:
            full_perm = tuple(permutation)
        return self.cos_err[perm_to_transition(full_perm, D=3)]

    # %% Permuted geometric theta and error
    def theta_act_perm(
        self, permutation: Iterable[int], start_point: int = 0
    ) -> np.ndarray:
        """The angles from a permutation [radians]"""
        return np.arccos(self.cos_act_perm(permutation, start_point=start_point))

    def theta_act_err_perm(
        self, permutation: Iterable[int], start_point: int = 0
    ) -> np.ndarray:
        """Standard error of theta"""
        if start_point is not None:
            full_perm = tuple([start_point] + list(permutation))
        else:
            full_perm = tuple(permutation)
        cos_ijk = self.cos_act[perm_to_transition(full_perm, D=3)]
        return self.cos_err[perm_to_transition(full_perm, D=3)] / np.sqrt(
            1 - cos_ijk**2
        )

    # %% Cached TANGO estimates and error
    @property
    def tango_estimates(self) -> np.ndarray:
        """Local incoming energy estimates"""
        # return phys.tango_incoming_estimate(
        #     self.energy_matrix[np.newaxis, :, np.newaxis], 1 - self.cos_act
        # )
        _calculator.set_event(self)
        return _calculator.tango_estimates((self.id, tuple(self.points[1].x)))

    @property
    def tango_partial_derivatives(self) -> Tuple[np.ndarray]:
        """d/de and d/d_cos for the TANGO estimates"""
        # return phys.partial_tango_incoming_derivatives(
        #     self.energy_matrix[np.newaxis, :, np.newaxis], 1 - self.cos_act
        # )
        _calculator.set_event(self)
        return _calculator.tango_partial_derivatives((self.id, tuple(self.points[1].x)))

    @property
    def tango_estimates_sigma(self) -> np.ndarray:
        """Error in local incoming energy estimates"""
        # eres = 1e-3
        # return np.sqrt(
        #     (eres * self.tango_partial_derivatives[0]) ** 2
        #     + (self.cos_err * self.tango_partial_derivatives[1]) ** 2
        # )
        # # return tango_incoming_sigma(self.energy_matrix[np.newaxis,:,np.newaxis],
        # #                             1 - self.cos_act,
        # #                             self.cos_err, eres=1e-3)
        _calculator.set_event(self)
        return _calculator.tango_estimates_sigma((self.id, tuple(self.points[1].x)))

    # %% Permuted incoming energy estimates
    def tango_estimates_perm(
        self, permutation: Iterable[int], start_point: int = 0
    ) -> np.ndarray:
        """Selected tango estimates"""
        full_perm = tuple([start_point] + list(permutation))
        return self.tango_estimates[perm_to_transition(full_perm)]

    def tango_estimates_sigma_perm(
        self, permutation: Iterable[int], start_point: int = 0
    ) -> np.ndarray:
        """Selected tango estimate standard error"""
        full_perm = tuple([start_point] + list(permutation))
        return self.tango_estimates_sigma[perm_to_transition(full_perm)]

    # %% Permuted TANGO incoming energy estimates
    def estimate_start_energy_perm(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        use_threshold: bool = False,
    ) -> float:
        """
        Given a permutation, return the TANGO incoming energy estimate

        If using a threshold, rejects the TANGO energy estimate and returns
        energy sum if non-physical
        """
        if start_point is not None:
            full_perm = tuple([start_point] + list(permutation))
        else:
            full_perm = tuple(permutation)
        energies = np.cumsum(self.energy_matrix[list(full_perm)])
        estimate = np.mean(
            energies[:-2] + self.tango_estimates[perm_to_transition(full_perm)]
        )
        if use_threshold:
            e_sum = self.energy_sum(permutation)
            if estimate < e_sum:
                return e_sum
            e_final = self.energy_matrix[permutation[-1]]
            estimated_outgoing = estimate - e_sum
            if phys.njit_cos_theor(estimated_outgoing + e_final, estimated_outgoing) < -1:
                return e_sum
        return estimate

    def estimate_start_energy_sigma_weighted_perm(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        eres=1e-3,
        use_threshold: bool = False,
    ) -> float:
        """
        Given a permutation, return the TANGO incoming energy estimate
        weighted by standard error

        If using a threshold, rejects the TANGO energy estimate and returns
        energy sum if non-physical
        """
        if start_point is not None:  # and start_point is not in permutation:?
            full_perm = tuple([start_point] + list(permutation))
        else:
            full_perm = tuple(permutation)
        energies = np.cumsum(self.energy_matrix[list(full_perm)])
        estimate = np.sum(
            (
                energies[:-2]
                + self.tango_estimates_perm(
                    permutation=permutation, start_point=start_point
                )
            )
            / (
                self.tango_estimates_sigma_perm(
                    permutation=permutation, start_point=start_point
                )
                + eres * np.sqrt(np.arange(1, len(energies) - 1, 1))
            )
        ) / np.sum(
            1
            / (
                self.tango_estimates_sigma_perm(
                    permutation=permutation, start_point=start_point
                )
                + eres * np.sqrt(np.arange(1, len(energies) - 1, 1))
            )
        )
        if use_threshold:
            e_sum = self.energy_sum(permutation)
            if estimate < e_sum:
                return e_sum
            e_final = self.energy_matrix[permutation[-1]]
            estimated_outgoing = estimate - e_sum
            if phys.njit_cos_theor(estimated_outgoing + e_final, estimated_outgoing) < -1:
                return e_sum
        return estimate

    def threshold_tango(
        self,
        permutation: Iterable[int],
        estimate: float,
    ):
        """Use the TANGO estimate if physical, otherwise use the energy sum"""
        e_sum = self.energy_sum(permutation)
        e_final = self.energy_matrix[permutation[-1]]
        estimated_outgoing = estimate - e_sum
        if phys.njit_cos_theor(estimated_outgoing + e_final, estimated_outgoing) < -1:
            return e_sum
        return estimate
        # if estimated_outgoing + e_final > compton_edge_incoming(estimated_outgoing):
        #     return e_sum
        # return estimate

    # %% Permuted theoretical cosine and error
    def cos_theor_perm(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
    ) -> np.ndarray:
        """Theoretical cosine"""
        energies = self.cumulative_energies(
            tuple(permutation), start_point=start_point, start_energy=start_energy
        )
        return phys.cos_theor_sequence(energies)

    def cos_theor_sigma_perm(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        Nmi: int = None,
    ) -> np.ndarray:
        """Theoretical cosine error"""
        energies = self.cumulative_energies(
            tuple(permutation), start_point=start_point, start_energy=start_energy
        )
        return phys.cos_theor_sigma(
            energies[:-1], energies[1:], Nmi,
        )

    # %% Permuted theoretical theta and error
    def theta_theor_perm(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        **theta_theor_kwargs,
    ) -> np.ndarray:
        """Theoretical cosine"""
        energies = self.cumulative_energies(
            tuple(permutation), start_point=start_point, start_energy=start_energy
        )
        return phys.theta_theor(energies[:-1], energies[1:], **theta_theor_kwargs)

    def theta_theor_sigma_perm(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        Nmi: int = None,
        **theta_theor_kwargs,
    ) -> np.ndarray:
        """Theoretical theta error"""
        return self.cos_theor_sigma_perm(
            permutation, start_point, start_energy, Nmi, **theta_theor_kwargs
        ) / np.sqrt(
            np.abs(
                1
                - self.cos_theor_perm(
                    permutation, start_point, start_energy, **theta_theor_kwargs
                )
                ** 2
            )
        )

    # %% Theoretical scattering energy from the CSF and error
    def scattered_energy(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
    ) -> np.ndarray[float]:
        """Scattered outgoing energy from CSF"""
        energies = self.cumulative_energies(
            tuple(permutation), start_point=start_point, start_energy=start_energy
        )[:-1]
        if start_point is not None:
            full_perm = tuple([start_point] + list(permutation))
        else:
            full_perm = tuple(permutation)
        return phys.outgoing_energy_csf(
            energies, 1 - self.cos_act[perm_to_transition(full_perm)]
        )

    def scattered_energy_sigma(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        Nmi: int = None,
        eres: float = 1e-3,
    ) -> np.ndarray[float]:
        """Scattered outgoing energy from CSF"""
        if Nmi is None:
            Nmi = len(permutation)
        energies = self.cumulative_energies(
            tuple(permutation), start_point=start_point, start_energy=start_energy
        )[:-1]
        return phys.outgoing_energy_csf_sigma(
            energies,
            1 - self.cos_act_perm(permutation, start_point),
            self.cos_act_err_perm(permutation, start_point),
            self.cumulative_energy_sigma(permutation, Nmi, eres)[:-1],
        )

    # %% Cumulative energy and error
    @lru_cache(maxsize=100)
    def cumulative_energies(
        self,
        permutation: Tuple[int],
        start_point: int = 0,
        start_energy: float = None,
    ) -> np.ndarray:
        """Get the cumulative energies along a permutation"""
        assert len(permutation) > 0, "Permutation must be length 1 or longer"
        if start_point is not None:
            full_perm = tuple([start_point] + list(permutation))
        else:
            full_perm = tuple(permutation)
        energies = self.energy_matrix[list(full_perm)]
        cum_energies = reverse_cumsum(energies)
        if start_energy is None:
            return cum_energies[1:]
        return cum_energies[1:] + (start_energy - cum_energies[1])

    def cumulative_energy_sigma(
        self, permutation: Iterable[int], Nmi: int = None, eres: float = 1e-3
    ) -> np.ndarray[float]:
        """Cumulative energy error"""
        if Nmi is None:
            Nmi = len(permutation)
        assert Nmi >= len(permutation)
        return eres * np.sqrt(np.arange(Nmi, Nmi - len(permutation), -1, dtype=int))

    # %% Residuals and residual standard error
    def res_sum_geo(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
    ) -> np.ndarray:
        """Get the residual between E_sum and E_geo"""
        e_sum = self.cumulative_energies(
            permutation=tuple(permutation),
            start_point=start_point,
            start_energy=start_energy,
        )[1:]
        e_scatter = self.scattered_energy(
            permutation=permutation, start_point=start_point, start_energy=start_energy
        )
        return e_sum - e_scatter

    def res_sum_geo_sigma(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        Nmi: int = None,
        eres: float = 1e-3,
        debug: bool = False,
    ) -> np.ndarray:
        """Get the standard error for the residual between E_sum and E_geo"""
        if Nmi is None:
            Nmi = len(permutation) + 1
        e_sum_lm1 = self.cumulative_energies(
            permutation=tuple(permutation),
            start_point=start_point,
            start_energy=start_energy,
        )[:-1]
        e_scatter = self.scattered_energy(
            permutation=permutation, start_point=start_point, start_energy=start_energy
        )
        err_cos = self.cos_act_err_perm(
            permutation=permutation, start_point=start_point
        )
        Ns = np.arange(Nmi, Nmi - len(e_sum_lm1), -1, dtype=int)
        if debug:  # Debug information
            print("Ns ", Ns)
            print(
                "a ", eres**2 * (Ns + 1) + (err_cos * (e_scatter**2 / phys.MEC2)) ** 2
            )
            print("b ", eres * (e_scatter / e_sum_lm1) ** 2)
            print(
                "delta_e_scatter ",
                eres * np.sqrt(Ns * (1 - (e_scatter / e_sum_lm1) ** 2)),
            )
            print("delta_e_scatter ", eres * np.sqrt(Ns))
            print("d ", (1 - (e_scatter / e_sum_lm1) ** 2))
            print("delta_e_scatter_n ", (err_cos * (e_scatter**2 / phys.MEC2)))

        # # Old error computation (does not include some error from energy)
        # return np.sqrt(eres**2 * Ns  + \
        #                (err_cos * (e_scatter**2/phys.MEC2))**2)
        return np.sqrt(
            eres**2
            * (
                (e_scatter / e_sum_lm1) ** 4
                + Ns * (1 - (e_scatter / e_sum_lm1) ** 2) ** 2
            )
            + (err_cos * (e_scatter**2 / phys.MEC2)) ** 2
        )

    def res_sum_loc(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
    ) -> np.ndarray:
        """Get the residual between E_sum and E_loc"""
        e_sum = self.cumulative_energies(
            permutation=tuple(permutation),
            start_point=start_point,
            start_energy=start_energy,
        )[:-1]
        e_loc = self.tango_estimates_perm(
            permutation=permutation, start_point=start_point
        )
        return e_sum - e_loc

    def res_sum_loc_sigma(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        Nmi: int = None,
        eres: float = 1e-3,
    ) -> np.ndarray:
        """Get standard error of the residual between E_sum and E_loc"""
        if Nmi is None:
            Nmi = len(permutation)
        Ns = np.arange(Nmi, Nmi - len(permutation), -1, dtype=int)[:-1] - 1
        full_perm = tuple([start_point] + list(permutation))
        err_cos = self.cos_act_err_perm(
            permutation=permutation, start_point=start_point
        )
        d_de = self.tango_partial_derivatives[0][perm_to_transition(full_perm)]
        (d_d_cos) = self.tango_partial_derivatives[1][perm_to_transition(full_perm)]
        return np.sqrt((eres**2 * ((1 - d_de) ** 2 + Ns)) + (err_cos * (d_d_cos)) ** 2)

    def res_loc_geo(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
    ) -> np.ndarray:
        """Get the residual between (E_loc - e) and E_geo"""
        e_loc_me = (
            self.tango_estimates_perm(permutation=permutation, start_point=start_point)
            - self.energy_matrix[list(permutation[:-1])]
        )
        e_scatter = self.scattered_energy(
            permutation=permutation, start_point=start_point, start_energy=start_energy
        )
        return e_loc_me - e_scatter

    def res_loc_geo_sigma(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        Nmi: int = None,
        eres: float = 1e-3,
    ) -> np.ndarray:
        """Standard error for residual between (E_loc - e) and E_geo residual"""
        # raise NotImplementedError
        if Nmi is None:
            Nmi = len(permutation)
        e_sum = self.cumulative_energies(
            permutation=tuple(permutation),
            start_point=start_point,
            start_energy=start_energy,
        )[1:]
        e_scatter = self.scattered_energy(
            permutation=permutation, start_point=start_point, start_energy=start_energy
        )
        err_cos = self.cos_act_err_perm(
            permutation=permutation, start_point=start_point
        )
        Ns = np.arange(Nmi, Nmi - len(e_sum), -1, dtype=int)
        full_perm = tuple([start_point] + list(permutation))
        d_de = self.tango_partial_derivatives[0][perm_to_transition(full_perm)] - 1
        d_d_cos = self.tango_partial_derivatives[1][perm_to_transition(full_perm)]
        return np.sqrt(
            eres**2
            * ((d_de - (e_scatter / e_sum) ** 2) ** 2 + Ns * (e_scatter / e_sum) ** 4)
            + err_cos**2 * (d_d_cos - e_scatter**2 / phys.MEC2) ** 2
        )

    def res_cos(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        **cos_theor_kwargs,
    ) -> np.ndarray:
        """Residual between geometric cosine and theoretical cosine"""
        geometric_cosine = self.cos_act_perm(permutation=permutation, start_point=start_point)
        theo = self.cos_theor_perm(
            permutation=permutation,
            start_point=start_point,
            start_energy=start_energy,
            **cos_theor_kwargs,
        )
        return geometric_cosine - theo

    def res_cos_cap(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        **cos_theor_kwargs,
    ) -> np.ndarray:
        """Residual between geometric cosine and theoretical cosine (capped at -1)"""
        geometric_cosine = self.cos_act_perm(
            permutation=permutation, start_point=start_point
        )
        theo = self.cos_theor_perm(
            permutation=permutation,
            start_point=start_point,
            start_energy=start_energy,
            **cos_theor_kwargs,
        )
        np.maximum(theo, -1, out=theo)
        return geometric_cosine - theo

    def res_cos_sigma(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        Nmi: int = None,
        eres: float = 1e-3,
    ) -> np.ndarray:
        """Standard error for residual between geometric cosine and theoretical
        cosine"""
        if Nmi is None:
            Nmi = len(permutation)
        err_cos = self.cos_act_err_perm(
            permutation=permutation, start_point=start_point
        )
        err_e = self.cos_theor_sigma_perm(
            permutation=permutation,
            start_point=start_point,
            start_energy=start_energy,
            Nmi=Nmi,
        )
        return np.sqrt(err_cos**2 + (eres * err_e) ** 2)

    def res_theta(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        fix_nan: Optional[float] = 2 * np.pi,
        **cos_theor_kwargs,
    ) -> np.ndarray:
        """Residual between geometric theta and theoretical theta"""
        geometric_theta = self.theta_act_perm(
            permutation=permutation, start_point=start_point
        )
        theo = self.theta_theor_perm(
            permutation=permutation,
            start_point=start_point,
            start_energy=start_energy,
            **cos_theor_kwargs,
        )
        out = geometric_theta - theo
        if fix_nan is not None:
            out[np.isnan(out)] = fix_nan
        return out

    def res_theta_cap(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        fix_nan: Optional[float] = 2 * np.pi,
        **cos_theor_kwargs,
    ) -> np.ndarray:
        """Residual between geometric theta and theoretical theta (cosine capped at -1)"""
        geometric_theta = self.theta_act_perm(
            permutation=permutation, start_point=start_point
        )
        theo = self.cos_theor_perm(
            permutation=permutation,
            start_point=start_point,
            start_energy=start_energy,
            **cos_theor_kwargs,
        )
        theo = np.arccos(np.maximum(theo, -1))
        out = geometric_theta - theo
        if fix_nan is not None:
            out[np.isnan(out)] = fix_nan
        return out

    def res_theta_sigma(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        Nmi: int = None,
        eres: float = 1e-3,
    ) -> np.ndarray:
        """Standard error for residual between geometric cosine and theoretical
        cosine"""
        if Nmi is None:
            Nmi = len(permutation)
        err_cos = self.theta_act_err_perm(
            permutation=permutation, start_point=start_point
        )
        err_e = self.theta_theor_sigma_perm(
            permutation=permutation,
            start_point=start_point,
            start_energy=start_energy,
            Nmi=Nmi,
        )
        return np.sqrt(err_cos**2 + (eres * err_e) ** 2)

    def compton_penalty(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        **cos_theor_kwargs,
    ) -> np.ndarray:
        """Indicator vector for a Compton Penalty"""
        theo = self.cos_theor_perm(
            permutation=permutation,
            start_point=start_point,
            start_energy=start_energy,
            **cos_theor_kwargs,
        )
        return phys.compton_penalty(theo)

    def compton_penalty_ell1(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        **cos_theor_kwargs,
    ) -> np.ndarray:
        """Indicator vector for a Compton Penalty"""
        theo = self.cos_theor_perm(
            permutation=permutation,
            start_point=start_point,
            start_energy=start_energy,
            **cos_theor_kwargs,
        )
        return phys.compton_penalty_ell1(theo)

    # %% Linear attenuation and cross-sections
    def linear_attenuation_abs(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
    ) -> np.ndarray[float]:
        """Absorption cross section in Germanium [cm^-1]"""
        energies = self.cumulative_energies(
            tuple(permutation), start_point=start_point, start_energy=start_energy
        )
        return phys.lin_att_abs(energies)

    def linear_attenuation_compt(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
    ) -> np.ndarray[float]:
        """Compton cross section in Germanium [cm^-1]"""
        energies = self.cumulative_energies(
            tuple(permutation), start_point=start_point, start_energy=start_energy
        )
        return phys.lin_att_compt(energies)

    def linear_attenuation_pair(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
    ) -> np.ndarray[float]:
        """Pair production cross section in Germanium [cm^-1]"""
        energies = self.cumulative_energies(
            tuple(permutation), start_point=start_point, start_energy=start_energy
        )
        return phys.lin_att_pair(energies)

    def lin_mu_total(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
    ) -> np.ndarray[float]:
        """Pair production cross section in Germanium [cm^-1]"""
        energies = self.cumulative_energies(
            tuple(permutation), start_point=start_point, start_energy=start_energy
        )
        return phys.lin_att_total(energies)

    def klein_nishina(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        use_ei: bool = True,
        **kwargs,
    ) -> np.ndarray[float]:
        """Klein-Nishina differential cross-section integrated about incoming ray axis"""
        energies = self.cumulative_energies(
            tuple(permutation), start_point=start_point, start_energy=start_energy
        )
        if use_ei:
            return phys.KN_differential_cross(
                energies[:-1],
                1 - self.cos_act_perm(permutation),
                energies[1:],
                integrate=True,
                **kwargs,
            )
        return phys.KN_differential_cross(
            energies[:-1], 1 - self.cos_act_perm(permutation), integrate=True, **kwargs
        )

    def klein_nishina_differential_cross_section(
        self,
        permutation: Iterable[int],
        start_point: int = 0,
        start_energy: float = None,
        use_ei: bool = True,
        **kwargs,
    ) -> np.ndarray[float]:
        """Klein-Nishina differential cross-section"""
        energies = self.cumulative_energies(
            tuple(permutation), start_point=start_point, start_energy=start_energy
        )
        if use_ei:
            return phys.KN_differential_cross(
                energies[:-1],
                1 - self.cos_act_perm(permutation),
                energies[1:],
                **kwargs,
            )
        return phys.KN_differential_cross(
            energies[:-1], 1 - self.cos_act_perm(permutation), **kwargs
        )

    # # %% Reduction
    # @cached_property
    # def quality_tensor(self):
    #     """
    #     Get transition quality tensor.
    #     """
    #     from greto.transition_grade_clustering import (
    #         get_grade_features,
    #     )  # pylint: disable=import-outside-toplevel

    #     f = get_grade_features(self)
    #     # TODO - implement a feature reduction here
    #     return np.sum(f, axis=-1)

    # @cached_property
    # def reduction(self):
    #     """
    #     from a degree three tensor to a
    #     degree two tensor
    #     """
    #     f = self.quality_tensor
    #     # TODO - implement a tensor reduction here
    #     return np.sum(f, axis=1)

    # %% Summary
    def summary(self, clusters: dict, true_energies: dict = None) -> None:
        """Print a summary of the event to standard output"""
        s = (
            f"Event {self.id} containing {len(clusters)} gamma rays"
            + f" from {len(self.hit_points)} detected interactions"
        )
        for cluster_id, cluster in clusters.items():
            s += f"\nGamma ray {cluster_id}:"
            s += f"\n  Detected energy {self.energy_sum(cluster):>2.4f} MeV"
            if true_energies is not None:
                s += f"\n      True energy {true_energies[cluster_id]/1000:>2.4f} MeV"
            for i, index in enumerate(cluster):
                s += f"\n   {i + 1} : {index} : {self.points[index]}"
        print(s)

    # %% Data management
    def flush_caches(self):
        """
        Delete cached property values and prepare for changes to properties
        """
        self.flush_position_caches()
        self.flush_energy_caches()

    def flush_position_caches(self):
        """
        Delete cached property values related to interaction positions and
        prepare for changes to properties
        """
        try:
            del self.point_matrix
        except AttributeError:
            pass
        try:
            del self.spherical_point_matrix
        except AttributeError:
            pass
        try:
            del self.data_matrix
        except AttributeError:
            pass
        try:
            del self.distance
        except AttributeError:
            pass
        try:
            del self.angle_distance
        except AttributeError:
            pass
        try:
            del self.ge_distance
        except AttributeError:
            pass
        try:
            del self.cos_act
        except AttributeError:
            pass
        try:
            del self.cos_err
        except AttributeError:
            pass
        try:
            del self.theta_err
        except AttributeError:
            pass
        try:
            del self.tango_estimates
        except AttributeError:
            pass
        try:
            del self.tango_partial_derivatives
        except AttributeError:
            pass
        try:
            del self.tango_estimates_sigma
        except AttributeError:
            pass

    def flush_energy_caches(self):
        """
        Delete cached property values related to interaction energies and
        prepare for changes to properties
        """
        try:
            del self.energy_matrix
        except AttributeError:
            pass
        try:
            del self.data_matrix
        except AttributeError:
            pass
        try:
            del self.position_uncertainty
        except AttributeError:
            pass
        try:
            del self.tango_estimates
        except AttributeError:
            pass
        try:
            del self.tango_partial_derivatives
        except AttributeError:
            pass
        try:
            del self.tango_estimates_sigma
        except AttributeError:
            pass

    def copy(self):
        """Duplicate the current event"""
        return Event(
            self.id,
            list(self.points),
            ground_truth=self.ground_truth,
            detector=self.detector_config,
            flat=self.flat,
        )

    # def subevent(self, cluster: tuple[int], cluster_id: int):
    #     return Event(
    #         (self.id, cluster_id), [self.points[i] for i in cluster], detector=self.detector_config
    #     )

    @property
    def coincidence(self):
        """Copy the event into a stripped down coincidence class"""
        return Coincidence(self.id, self.hit_points, self.detector_config.detector)

    # def remove_zero_energy_interactions(self, threshold:float = 0.,
    #                                     clusters:Dict[int]=None):
    #     """Remove any interactions with zero energy"""
    #     keep_indices = [i for i, p in enumerate(self.points) if i == 0 or p.e > 0.]
    #     new_event = Event(self.id, [self.hit_points[i] for i in keep_indices])
    #     if clusters is not None:
    #         new_clusters = {}
    #         for cluster_id, cluster in clusters.items():
    #             for id in cluster:
    #                 if id not in keep_indices:


# %% Test function
# def outer_test_func(event:Event, radius=None):
#     """Test function for attribute assignment"""
#     if radius is None:
#         radius = default_config.inner_radius
#     print(event)
#     print(radius)
#     print(default_config.inner_radius)

# setattr(Event, 'outer_test_func', outer_test_func)


# %% Event Calculator


class _EventCalculator:
    """
    Calculator for event computations

    Separate from the Event for memory purposes
    """

    def __init__(self) -> None:
        self.event:Event = None
        self.event_id = None

    def set_event(self, event: Event, cluster_id: Optional[int] = None) -> None:
        """Set the event (and cluster id) for calculations"""
        self.event = event
        if cluster_id is not None:
            self.event_id = ((event.id, cluster_id), tuple(event.points[1].x))
        else:
            self.event_id = (event.id, tuple(event.points[1].x))

    @lru_cache(maxsize=100)
    def distance(self, event_id: int | tuple):
        """Cached distance using event_id as the caching key"""
        # return squareform(geo.pairwise_distance(self.event.point_matrix))
        return geo.njit_square_pdist(self.event.point_matrix)

    # def distance_perm(
    #     self, permutation: tuple[int], start_point: int = 0
    # ) -> np.ndarray:
    #     """Distances between points in a permutation"""
    #     if start_point is not None:
    #         full_perm = tuple([start_point] + list(permutation))
    #     else:
    #         full_perm = tuple(permutation)
    #     return self.distance(self.event_id)[perm_to_transition(full_perm, D=2)]

    @lru_cache(maxsize=100)
    def angle_distance(self, event_id: int | tuple) -> np.ndarray:
        """
        Angular distance between two points
        TODO - can use cos_act property for this to avoid possible extra computation
        """
        # return squareform(np.arccos(1.0 - pdist(self.event.hit_point_matrix, metric="cosine")))
        return np.arccos(1.0 - geo.njit_square_cosine_pdist(self.event.hit_point_matrix))

    @lru_cache(maxsize=100)
    def ge_distance(self, event_id: int | tuple) -> np.ndarray:
        """Distance between two points"""
        # return squareform(geo.ge_distance(self.event.point_matrix,d12_euc=squareform(self.distance(self.event_id)),))
        return geo.njit_square_ge_pdist(self.event.point_matrix, self.event.detector_config.inner_radius, d12_euc=self.distance(self.event_id))

    # def ge_distance_perm(
    #     self, permutation: Iterable[int], start_point: int = 0
    # ) -> np.ndarray:
    #     """Distances between points in a permutation"""
    #     if start_point is not None:
    #         full_perm = tuple([start_point] + list(permutation))
    #     else:
    #         full_perm = tuple(permutation)
    #     return self.ge_distance(self.event_id)[perm_to_transition(full_perm, D=2)]

    # def linkage_array(
    #     self,
    #     distance: str = "great_circle",
    #     method: str = "single",
    #     time_gap: int = 40,
    #     center_cross_penalty: float = 0.0,
    #     center_cross_threshold: float = 1e-3,
    #     center_cross_factor: float = 0.0,
    #     **kwargs,
    # ) -> np.ndarray:
    #     """Clustering linkage"""
    #     if len(self.event.hit_points) <= 1:
    #         return None
    #     time_gap_accept = (
    #         pdist(np.expand_dims(np.array([p.ts for p in self.event.hit_points]), axis=1))
    #         < time_gap
    #     )
    #     if distance.lower() == "euclidean":
    #         distances = squareform(self.distance(self.event_id)[1:, 1:])
    #     elif distance.lower() in ["great_circle", "cosine"]:
    #         distances = np.arccos(1 - pdist(self.event.hit_point_matrix, metric="cosine"))
    #     elif distance.lower() == "germanium":
    #         distances = squareform(self.ge_distance(self.event_id)[1:, 1:])
    #         if center_cross_penalty > 0.0 or center_cross_factor > 0.0:
    #             center_cross_factor = np.nan_to_num(center_cross_factor)
    #             center_cross_penalty = np.nan_to_num(center_cross_penalty)
    #             euc_distances = squareform(self.distance(self.event_id)[1:, 1:])
    #             if center_cross_factor > 0.0:
    #                 distances += (
    #                     np.maximum(euc_distances - distances, 0) * center_cross_factor
    #                 )
    #             if center_cross_penalty > 0.0:
    #                 distances[
    #                     euc_distances - distances > center_cross_threshold
    #                 ] += center_cross_penalty
    #     else:
    #         raise NotImplementedError

    #     distances[~time_gap_accept] += np.nan_to_num(np.inf)

    #     if method.startswith("dir") or method.startswith("asym"):
    #         Z = asym_hier_linkage(squareform(distances), **kwargs)
    #     else:
    #         Z = linkage(distances, method=method)
    #     self.linkage = Z
    #     return Z

    @lru_cache(maxsize=100)
    def cos_act(self, event_id: int | tuple) -> np.ndarray:
        """Cosine between interaction points"""
        return geo.cosine_ijk(self.event.point_matrix)

    @lru_cache(maxsize=100)
    def cos_err(self, event_id: int | tuple) -> np.ndarray:
        """Error in cosine due to position uncertainty"""
        err = geo.err_cos_vec_precalc(
            self.distance(self.event_id),
            self.cos_act(self.event_id),
            self.event.position_uncertainty,
        )
        # TODO - should the center point be made more accurate here or in position_uncertainty?
        # err[0,:,:] /= 2 # Center point is more accurate
        return err

    @lru_cache(maxsize=100)
    def theta_err(self, event_id: int | tuple) -> np.ndarray:
        """Error in cosine due to position uncertainty"""
        err = geo.err_theta_vec_precalc(
            self.distance(self.event_id),
            self.cos_act(self.event_id),
            self.event.position_uncertainty,
        )
        # TODO - should the center point be made more accurate here or in position_uncertainty?
        # err[0,:,:] /= 2 # Center point is more accurate as in AFT
        return err

    # def cos_act_perm(self, permutation: tuple[int], start_point: int = 0) -> np.ndarray:
    #     """The cosines of angles from permutation"""
    #     if start_point is not None:
    #         full_perm = tuple([start_point] + list(permutation))
    #     else:
    #         full_perm = tuple(permutation)
    #     return self.cos_act(self.event_id)[perm_to_transition(full_perm, D=3)]

    # def cos_act_err_perm(
    #     self, permutation: tuple[int], start_point: int = 0
    # ) -> np.ndarray:
    #     """Standard error of cosine"""
    #     if start_point is not None:
    #         full_perm = tuple([start_point] + list(permutation))
    #     else:
    #         full_perm = tuple(permutation)
    #     return self.cos_err(self.event_id)[perm_to_transition(full_perm, D=3)]

    # def theta_act_perm(
    #     self, permutation: Iterable[int], start_point: int = 0
    # ) -> np.ndarray:
    #     """The angles from a permutation [radians]"""
    #     return np.arccos(self.cos_act_perm(permutation, start_point=start_point))

    # def theta_act_err_perm(
    #     self, permutation: Iterable[int], start_point: int = 0
    # ) -> np.ndarray:
    #     """Standard error of theta"""
    #     if start_point is not None:
    #         full_perm = tuple([start_point] + list(permutation))
    #     else:
    #         full_perm = tuple(permutation)
    #     cos_ijk = self.cos_act(self.event_id)[perm_to_transition(full_perm, D=3)]
    #     return self.cos_err(self.event_id)[
    #         perm_to_transition(full_perm, D=3)
    #     ] / np.sqrt(1 - cos_ijk**2)

    @lru_cache(maxsize=100)
    def tango_estimates(self, event_id: int | tuple) -> np.ndarray:
        """Local incoming energy estimates"""
        return phys.tango_incoming_estimate(
            # self.event.energy_matrix[np.newaxis, :, np.newaxis],
            self.event.energy_matrix,
            1 - self.cos_act(self.event_id),
        )

    @lru_cache(maxsize=100)
    def tango_partial_derivatives(self, event_id: int | tuple) -> Tuple[np.ndarray]:
        """d/de and d/d_cos for the TANGO estimates"""
        return phys.partial_tango_incoming_derivatives(
            # self.event.energy_matrix[np.newaxis, :, np.newaxis],
            self.event.energy_matrix,
            1 - self.cos_act(self.event_id),
        )

    @lru_cache(maxsize=100)
    def tango_estimates_sigma(self, event_id: int | tuple) -> np.ndarray:
        """Error in local incoming energy estimates"""
        eres = 1e-3
        return np.sqrt(
            (eres * self.tango_partial_derivatives(self.event_id)[0]) ** 2
            + (
                self.cos_err(self.event_id)
                * self.tango_partial_derivatives(self.event_id)[1]
            )
            ** 2
        )


_calculator = _EventCalculator()
