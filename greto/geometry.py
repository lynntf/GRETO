"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Geometrical computations using interaction positions
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation

from greto import default_config
from greto.detector_config_class import DetectorConfig


def cos_act(
    p1: np.ndarray[float], p2: np.ndarray[float], p3: np.ndarray[float]
) -> float:
    """
    # Compute the cosine of angle formed by p1->p2->p3 using standard angle
    formula

    ## Args:
        - `p1`: The first interaction point coordinates
        - `p2`: The second interaction point coordinates
        - `p3`: The third interaction point coordinates
    """
    y1 = p2 - p1
    y2 = p3 - p2
    cos_theta_act = (y1 @ y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))

    # This will only ever be outside of (-1, 1) if there are rounding errors
    cos_theta_act = np.clip(cos_theta_act, -1.0, 1.0)
    return cos_theta_act


def theta_act(
    p1: np.ndarray[float], p2: np.ndarray[float], p3: np.ndarray[float]
) -> float:
    """
    # Compute the angle formed by p1->p2->p3 using standard angle formula

    ## Args:
        - `p1`: The first interaction point coordinates
        - `p2`: The second interaction point coordinates
        - `p3`: The third interaction point coordinates
    """
    return np.arccos(cos_act(p1, p2, p3))


def condensed_index(m: int, i: int, j: int) -> int:
    """
    # Return the condensed index for a condensed pairwise distance matrix

    ## Args:
        - `m`: The number of original observations
        - `i`: The first index
        - `j`: The second index
    """
    return m * i + j - ((i + 2) * (i + 1)) // 2


def pairwise_distance(points: np.ndarray[float]) -> np.ndarray[float]:
    """
    # Returns the condensed pairwise (Euclidean) distance between the array of
    point coordinates

    ## Args:
        - `points`: Interaction point coordinates
    """
    return pdist(points, metric="euclidean")


def cosine_distance(points: np.ndarray[float]) -> np.ndarray[float]:
    """
    # Returns the condensed pairwise (1 - cos theta) distance between the array
    of point coordinates

    ## Args:
        - `points`: Interaction point coordinates
    """
    return pdist(points, metric="cosine")


def cosine_vec(
    points: np.ndarray[float], center: np.ndarray = np.array([0.0, 0.0, 0.0])
) -> np.ndarray[float]:
    """
    Returns the pairwise cosine between vectors center -> point_1 and center ->
    point_2.

    If we want the cosine between point_1 -> center and center -> point_2, this
    is negative. This allows the computation of angles centered away from the
    origin.

    pdist will return a 1 - cosine distance

    Args:
        - points: Interaction point coordinates
        - center: Center of the angle, default is the origin
    Returns:
        - cosine distance between points given a center point
    """
    return 1.0 - pdist(points - center, metric="cosine")


def one_minus_cosine_ijk(points: np.ndarray[float]) -> np.ndarray[float]:
    """
    Get all of the 1 - cos theta values for all transitions i -> j -> k

    Args:
        - points: Interaction point coordinates
    Returns:
        - array of 1 - cos values for transitions
    """
    N = len(points)
    o_m_cosine_ijk = np.zeros((N, N, N))
    for j in range(0, N):
        # omc_ijk = 1 + cosine_vec(points, points[j])
        # omc_ijk = 2.0 - pdist(points - points[j], metric="cosine")
        # o_m_cosine_ijk[:, j, :] = squareform(omc_ijk)
        o_m_cosine_ijk[:, j, :] = squareform(2.0 - pdist(points - points[j], metric="cosine"))
    o_m_cosine_ijk[
        np.logical_or(np.isnan(o_m_cosine_ijk), ~np.isfinite(o_m_cosine_ijk))
    ] = 0.0
    return o_m_cosine_ijk


def err_cos_vec(
    points: np.ndarray[float], position_err: np.ndarray[float] = 0.5
) -> np.ndarray[float]:
    """
    # Error propagation in the computation of the cos(theta).

    From the definition of the dot product:
    cos(theta) = dot((b-a), (c-b))/(norm(b-a)*norm(c-b))

    ## Args:
        - `points`: Interaction point coordinates
        - `position_err`: Error in position values [cm] (AGATA is 8mm,
            GRETA/GRETINA is 5mm)
    ## Returns:
        - `error`: Propagated error
    """
    distances = squareform(pairwise_distance(points))
    cos_ijk = 1 - one_minus_cosine_ijk(points)
    return err_cos_vec_precalc(distances, cos_ijk, position_err)


def partial_err_theta_vec_precalc(
    distances: np.ndarray[float],
    cos_ijk: np.ndarray[float],
    position_err: np.ndarray[float] = 0.5,
) -> Tuple[np.ndarray[float]]:
    """
    # Partial error propagation in the computation of the theta.

    From the definition of the dot product:
    cos(theta) = dot((b-a), (c-b))/(norm(b-a)*norm(c-b))

    ## Args:
        - `distances` : Distances (Euclidean) between interaction points
        - `position_err`: Error in position values [cm] (AGATA is 8mm,
            GRETA/GRETINA is 5mm)
    ## Returns:
        - Errors as a separate function of the three positions involved in the
          computation
    """
    rab = np.copy(distances[:, :, np.newaxis])
    rbc = np.copy(distances[np.newaxis, :, :])
    rab[rab <= 0.0] = 1
    rbc[rbc <= 0.0] = 1
    sigma_squared_a = np.zeros(cos_ijk.shape)
    sq_position_err = position_err**2
    if isinstance(sq_position_err, float):
        sigma_squared_a = 2 * sq_position_err / (rab**2)
        sigma_squared_b = 2 * sq_position_err * cos_ijk / (rab * rbc)
        sigma_squared_c = 2 * sq_position_err / (rbc**2)
    else:
        sigma_squared_a = (
            sq_position_err[:, np.newaxis, np.newaxis]
            + sq_position_err[np.newaxis, :, np.newaxis]
        ) / (rab**2)
        sigma_squared_b = (
            2 * sq_position_err[np.newaxis, :, np.newaxis] * (cos_ijk) / (rab * rbc)
        )
        sigma_squared_c = (
            sq_position_err[np.newaxis, :, np.newaxis]
            + sq_position_err[np.newaxis, np.newaxis, :]
        ) / (rbc**2)
    return (sigma_squared_a, sigma_squared_b, sigma_squared_c)


def err_cos_vec_precalc(
    distances: np.ndarray[float],
    cos_ijk: np.ndarray[float],
    position_err: np.ndarray[float] = 0.5,
) -> np.ndarray[float]:
    """
    # Error propagation in the computation of the cos(theta).

    From the definition of the dot product:
    cos(theta) = dot((b-a), (c-b))/(norm(b-a)*norm(c-b))

    ## Args:
        - `distances` : Distances (Euclidean) between interaction points
        - `cos_ijk` : Cosine values for all angles
        - `position_err` : Error in position values [cm] (AGATA is 8mm,
            GRETA/GRETINA is 5mm)
    ## Returns:
        - Propagated error
    """
    sa, sb, sc = partial_err_theta_vec_precalc(
        distances=distances, cos_ijk=cos_ijk, position_err=position_err
    )
    return np.sqrt((1 - np.square(cos_ijk)) * (sa + sb + sc))


def err_theta_vec_precalc(
    distances: np.ndarray[float],
    cos_ijk: np.ndarray[float],
    position_err: np.ndarray[float] = 0.5,
) -> np.ndarray[float]:
    """
    # Error propagation in the computation of the theta.

    From the definition of the dot product:
    theta = arccos(dot((b-a), (c-b))/(norm(b-a)*norm(c-b)))

    ## Args:
        - `distances` : Distances (Euclidean) between interaction points
        - `cos_ijk` : Cosine values for all angles
        - `position_err` : Error in position values [cm] (AGATA is 8mm,
            GRETA/GRETINA is 5mm)
    ## Returns:
        - Propagated error
    """
    sa, sb, sc = partial_err_theta_vec_precalc(
        distances=distances, cos_ijk=cos_ijk, position_err=position_err
    )
    return np.sqrt((sa + sb + sc))


def ge_distance(
    points: np.ndarray[float],
    inner_radius: float = None,
    d12_euc: np.ndarray = None,
    detector: DetectorConfig = default_config,
) -> np.ndarray[float]:
    """
    # Vectorized distance through germanium

    This is the distance between interactions ignoring the hollow center of the
    detector shell. Returns condensed pairwise distance just like `scipy.pdist`

    ## Args:
        - `points`: Interaction point coordinates
        - `inner_radius`: Inner radius of the detector
        - `d12_euc`: Pairwise distance (Euclidean) between interactions
    ## Returns:
        - Germanium pairwise distance (condensed pairwise matrix)
    """
    if inner_radius is None:
        inner_radius = detector.inner_radius
    m = points.shape[0]
    if d12_euc is None:
        d12_euc = pdist(points, metric="euclidean")
    iis = np.zeros(d12_euc.shape[0], dtype=int)
    jjs = np.zeros(d12_euc.shape[0], dtype=int)
    for i in range(m - 1):
        for j in range(i + 1, m):
            iis[m * i + j - ((i + 2) * (i + 1)) // 2] = i
            jjs[m * i + j - ((i + 2) * (i + 1)) // 2] = j
    gamma = (points[jjs] - points[iis]) / d12_euc[:, np.newaxis]
    d1 = np.sum(points[iis] * (-gamma), axis=1)
    d_squared = (-d1) ** 2 - np.linalg.norm(points[iis], axis=1) ** 2 + inner_radius**2
    indicator = d_squared >= 0
    d = np.zeros(d_squared.shape)
    np.sqrt(d_squared, where=indicator, out=d)

    d2 = np.sum(points[jjs] * gamma, axis=1)
    lambda_1 = d1 - d
    lambda_2 = d2 - d
    ind = np.logical_and.reduce(
        (
            indicator,
            0 <= lambda_1,
            0 <= lambda_2,
            lambda_1 <= d12_euc,
            lambda_2 <= d12_euc,
        )
    )
    np.add(d12_euc, -2 * d, where=ind, out=d12_euc)
    ind = np.logical_and.reduce(
        (indicator, 0 <= lambda_1, lambda_2 < 0, lambda_1 <= d12_euc)
    )
    np.add(d12_euc, -d - d2, where=ind, out=d12_euc)
    ind = np.logical_and.reduce(
        (indicator, lambda_1 < 0, 0 <= lambda_2, lambda_2 <= d12_euc)
    )
    np.add(d12_euc, -d - d1, where=ind, out=d12_euc)
    ind = np.logical_and.reduce((indicator, lambda_1 < 0, lambda_2 < 0))
    d12_euc[ind] = 0.0
    return d12_euc


def germanium_extension(
    p1: np.ndarray[float],
    p2: np.ndarray[float],
    inner_radius: float = None,
    detector: DetectorConfig = default_config,
) -> np.ndarray[float]:
    """
    # Extend the distance between interactions to match a specified germanium
    distance

    Find the point in the direction `dir` from `p1` with Germanium distance
    `dist`. Assumes that `p1` is in the germanium and `p2` is separated from
    `p1` by an air gap.

    ## Args:
        - `p1` : First point
        - `direction` : Direction from first point
        - `dist` : Germanium distance to match
        - `inner_radius` : Radius of the inner wall of the detector shell
    ## Returns:
        - New point coordinates with the specified germanium distance
    """
    dist = np.linalg.norm(p2 - p1)
    direction = (p2 - p1) / dist
    if inner_radius is None:
        inner_radius = detector.inner_radius
    half_width = np.sqrt(
        inner_radius**2 - np.linalg.norm(p1 - direction * np.dot(p1, direction)) ** 2
    )
    d1 = abs(np.dot(p1, direction)) - half_width
    extra = 0
    if dist > d1:
        extra = 2 * half_width
        dist += 2 * half_width
    return extra * direction
    # return p1 + direction*(dist)


def cartesian_to_spherical(xyz: np.ndarray) -> np.ndarray:
    """Adapted from
    https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    Returns [[r, phi (elevation), theta (azimuthal)],...]
    """
    rpt = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    rpt[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)  # r
    # rpt[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    rpt[:, 1] = -np.arctan2(
        xyz[:, 2], np.sqrt(xy)
    )  # for elevation angle defined from XY-plane up
    rpt[:, 2] = np.mod(np.pi + np.arctan2(xyz[:, 1], xyz[:, 0]), 2 * np.pi) - np.pi
    return rpt


# %% Cone and sphere intersections


class cone:
    """Class for defining a cone and its intersections with a sphere"""

    def __init__(
        self,
        apex: np.ndarray,
        direction: np.ndarray,
        opening_angle: float,
        sphere_radius: float,
    ):
        """
        Initializes a cone with the given apex, direction, and opening angle.

        Parameters:
            apex (np.ndarray): The coordinates of the apex of the cone.
            direction (np.ndarray): The direction vector of the cone's axis.
            opening_angle (float): The opening angle of the cone in radians.

        Returns:
            None
        """
        self.apex = np.array(apex)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.opening_angle = opening_angle
        self._get_first_ray()
        self.ray_directions = None
        self.sphere_radius = sphere_radius
        self.squared_dist_apex_to_sphere = (
            np.dot(self.apex, self.apex) - sphere_radius**2
        )
        self.apex_magnitude = np.sqrt(np.dot(self.apex, self.apex))
        self.beta = np.arccos(
            np.dot(self.apex, self.direction) / self.apex_magnitude
        )  # angle between cone apex and cone axis
        self.c1 = np.sin(self.beta) * np.sin(self.opening_angle)
        self.c2 = np.cos(self.beta) * np.cos(self.opening_angle)

    def ray_lengths(self, theta: Iterable) -> np.ndarray:
        """
        Trig method for computing ray lengths. Theta here does not necessarily
        match the same directions as ray construction. By construction, theta=0
        should provide a minimum distance and theta=pi a maximum distance.

        Derivation notes:
        Spherical law of cosines:
        $A = \\theta$
        $a = \\phi$ (angle between orientation by theta and apex)
        $b = \\beta$ (angle from apex to direction)
        $c = \\alpha$ opening angle
        $\\cos(a) = \\cos(b)\\cos(c) + \\sin(b)\\sin(c)\\cos(A)$
        $\\cos(\\phi) = \\cos(\\beta) \\cos(\\alpha) + \\sin(\\beta) \\sin(\\alpha) \\cos(\\theta)

        Given $\\cos(\\phi)$, we can then get the distance from the apex to the sphere, $x$
        Make a triangle with $x \\sin(\\phi)$ by $x \\cos(\\phi)$.
        $\\|a\\| + x \\cos(\\phi)$ (apex length is $\\|a\\|$) is one side,
        $x \\sin(\\phi)$ is another, and $r$ is the third. Solve using quadratic formula:
        $x = -\\cos(\\phi) \\|a\\| + \\sqrt{ \\|a\\|^2 (\\cos^2(\\phi) - 1) + r^2 }$

        Parameters:
            theta:  angles about cone where theta=0 is oriented to the smallest
                distance and theta=pi is oriented to the largest distance (not
                oriented with coordinate system)
        Returns:
            np.ndarray:  distances from the cone apex to the surrounding sphere intersection
        """
        cos_phi = self.c1 * np.cos(theta) + self.c2
        lengths = -self.apex_magnitude * cos_phi + np.sqrt(
            self.apex_magnitude**2 * (cos_phi**2 - 1) + self.sphere_radius**2
        )
        return lengths

    def _get_first_ray(self):
        """The first ray is the one with the shortest distance to the sphere
        surface. Unit vector."""
        first_axis = np.cross(self.direction, self.apex)
        i = 0
        other_vector = np.zeros(first_axis.shape)
        # colinear, still need a perpendicular vector, cross with any other vector
        while np.linalg.norm(first_axis) == 0:
            other_vector[i] = 1.0
            first_axis = np.cross(self.direction, other_vector)
            i += 1
        first_axis /= np.linalg.norm(first_axis)
        first_ray = Rotation.from_rotvec(self.opening_angle * first_axis).apply(
            self.direction
        )
        self.first_ray = first_ray

    def get_ray_length(self, theta: float):
        """Reduced computation method for a single ray as a function of theta"""
        ray_direction = Rotation.from_rotvec(theta * self.direction).apply(
            self.first_ray
        )
        b = np.dot(2 * ray_direction, self.apex)
        # c = np.dot(self.apex, self.apex) - self.sphere_radius ** 2
        # length = (-b + np.sqrt(np.square(b) - 4 * c)) / (2)
        # This is just the quadratic formula:
        length = (-b + np.sqrt(np.square(b) - 4 * self.squared_dist_apex_to_sphere)) / (
            2
        )
        return length

    def get_thetas(self, num_rays: int) -> np.ndarray:
        """Returns an array of angles evenly spaced around the unit circle."""
        return np.linspace(0, 2 * np.pi, num_rays)

    def get_rays(self, num_rays: int) -> np.ndarray:
        """
        Generates directional vectors for rays originating from the cone's apex.

        The ray directions are obtained by taking the cone direction, rotating
        it by the opening angle (about any axis orthogonal to the cone
        direction), and then taking this single rotated ray and further rotating
        it about the cone direction (axis). The chosen orthogonal axis places
        the first ray in the direction of the nearest intersection with the
        sphere

        Parameters:
            num_rays (int): The number of rays to generate.

        Returns:
            np.ndarray: An array of ray direction vectors.
        """
        thetas = self.get_thetas(num_rays)
        ray_directions = Rotation.from_rotvec(
            thetas[:, np.newaxis] * self.direction[np.newaxis, :]
        ).apply(self.first_ray)
        ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]
        self.ray_directions = ray_directions
        return self.ray_directions

    def get_ray_lengths(self, num_rays):
        """
        Computes the lengths of the rays from the cone's apex to the sphere.

        Lengths come from the solution of a quadratic equation.

        Parameters:
            num_rays (int): The number of rays used to approximate the intersection.
            radius (float): The radius of the sphere.

        Returns:
            np.ndarray: An array of ray lengths.
        """
        ray_directions = self.get_rays(num_rays)
        a = np.linalg.norm(ray_directions, axis=1)
        b = np.dot(2 * ray_directions, self.apex)
        c = np.dot(self.apex, self.apex) - self.sphere_radius**2
        lengths = (-b + np.sqrt(np.square(b) - 4 * a * c)) / (2 * a)
        return lengths

    def get_intersections(self, num_rays):
        """
        Calculates the intersection points between the cone and the sphere.

        Parameters:
            num_rays (int): The number of rays used to approximate the intersection.
            radius (float): The radius of the sphere.

        Returns:
            np.ndarray: An array of intersection points.
        """
        lengths = self.get_ray_lengths(num_rays)  # also sets self.ray_directions
        # lengths = self.ray_lengths(self.get_thetas(num_rays))
        intersections = self.apex + lengths[:, np.newaxis] * self.ray_directions
        return intersections


def cone_ray_lengths(
    apex: np.ndarray,
    direction: np.ndarray,
    opening_angle: float,
    num_rays: int,
    radius: float,
) -> np.ndarray:
    """
    Calculates the lengths of rays from the cone's apex to the sphere.

    Parameters:
        apex (np.ndarray): The coordinates of the apex of the cone.
        direction (np.ndarray): The direction vector of the cone's axis.
        opening_angle (float): The opening angle of the cone in radians.
        num_rays (int): The number of rays used to approximate the intersection.
        radius (float): The radius of the sphere.

    Returns:
        np.ndarray: An array of ray lengths.
    """
    apex = np.array(apex)
    direction = np.array(direction) / np.linalg.norm(direction)
    thetas = 2 * np.pi * np.linspace(0, 1, num_rays, endpoint=False)

    first_axis = np.cross(direction, apex)
    i = 0
    other_vector = np.zeros(first_axis.shape)
    # colinear, still need a perpendicular vector, cross with any other vector
    while np.linalg.norm(first_axis) == 0:
        other_vector[i] = 1.0
        first_axis = np.cross(direction, other_vector)
        i += 1
    first_axis /= np.linalg.norm(first_axis)
    first_ray = Rotation.from_rotvec(opening_angle * first_axis).apply(direction)
    ray_directions = Rotation.from_rotvec(
        thetas[:, np.newaxis] * direction[np.newaxis, :]
    ).apply(first_ray)
    # ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]

    # a = np.linalg.norm(ray_directions, axis=1)
    a = 1.0
    b = np.dot(2 * ray_directions, apex)
    c = np.dot(apex, apex) - radius**2
    lengths = (-b + np.sqrt(np.square(b) - 4 * a * c)) / (2 * a)
    return lengths


@lru_cache(1)  # cache the latest value, which is the most likely to be reused
def cone_get_first_ray(apex: tuple, direction: tuple, opening_angle: float):
    """Get first ray for cone lengths"""
    apex = np.array(apex)
    direction = np.array(direction) / np.linalg.norm(direction)

    first_axis = np.cross(direction, apex)
    i = 0
    other_vector = np.zeros(first_axis.shape)
    # colinear, still need a perpendicular vector, cross with any other vector
    while np.linalg.norm(first_axis) == 0:
        other_vector[i] = 1.0
        first_axis = np.cross(direction, other_vector)
        i += 1
    first_axis /= np.linalg.norm(first_axis)
    first_ray = Rotation.from_rotvec(opening_angle * first_axis).apply(direction)
    return first_ray


def cone_ray_length(
    apex: np.ndarray,
    direction: np.ndarray,
    opening_angle: float,
    theta: float,
    radius: float,
) -> np.ndarray:
    """
    Calculates the length of a single ray from the cone's apex to the sphere.

    Parameters:
        apex (np.ndarray): The coordinates of the apex of the cone.
        direction (np.ndarray): The direction vector of the cone's axis.
        opening_angle (float): The opening angle of the cone in radians.
        num_rays (int): The number of rays used to approximate the intersection.
        radius (float): The radius of the sphere.

    Returns:
        np.ndarray: An array of ray lengths.
    """
    first_ray = cone_get_first_ray(tuple(apex), tuple(direction), opening_angle)
    ray_direction = Rotation.from_rotvec(theta * direction).apply(first_ray)

    b = np.dot(2 * ray_direction, apex)
    c = np.dot(apex, apex) - radius**2
    length = (-b + np.sqrt(np.square(b) - 4 * c)) / (2)
    return length


def cone_ray_length_2(
    apex: np.ndarray,
    direction: np.ndarray,
    opening_angle: float,
    theta: float,
    sphere_radius: float,
) -> np.ndarray:
    """
    Initializes a cone with the given apex, direction, and opening angle.

    Parameters:
        apex (np.ndarray): The coordinates of the apex of the cone.
        direction (np.ndarray): The direction vector of the cone's axis.
        opening_angle (float): The opening angle of the cone in radians.

    Returns:
        None
    """
    mag_a = np.sqrt(np.dot(apex, apex))
    beta = np.arccos(np.dot(apex, direction) / mag_a)  # angle between apex and axis
    c1 = np.sin(beta) * np.sin(opening_angle)
    c2 = np.cos(beta) * np.cos(opening_angle)

    cos_phi = c1 * np.cos(theta) + c2
    t = -mag_a * cos_phi + np.sqrt(mag_a**2 * (cos_phi**2 - 1) + sphere_radius**2)
    return t


def crystal_depth(global_coords: np.ndarray, crystal_coords: np.ndarray):
    """
    Get the depth into the crystal material under the assumption of a flat
    crystal surface (instead of the typical spherical assumption).

    Note that the depths using this method are very similar to the spherical
    assumption.
    """
    r_from_target = np.linalg.norm(global_coords, axis=-1)
    r_from_target = np.maximum(r_from_target, 1e-10)
    r_from_crystal_axis = np.linalg.norm(crystal_coords[:, 0:2], axis=-1)
    z_crystal = crystal_coords[:, 2]
    c_depth = (
        r_from_target * z_crystal / np.sqrt(r_from_target**2 - r_from_crystal_axis**2)
    )
    return c_depth
