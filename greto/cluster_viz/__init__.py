"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Methods for visualizing clusters of gamma ray interactions
"""
from __future__ import annotations

import math
from typing import Dict, List

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import percentileofscore

from greto import cluster_utils, default_config
from greto.detector_config_class import DetectorConfig
from greto.event_class import Event
from greto.event_tools import flatten_event

# from matplotlib import cm
# colors = cm.Set1.colors
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# %% 2D plots
def plot_points(event: Event, axis: plt.Axes = None, with_labels: bool = False):
    """Plot points in 2D using a Sinusoidal projection"""
    lats = []
    lons = []
    for point in event.hit_points:
        lat = 180 / np.pi * point.theta - 90
        lon = 180 / np.pi * point.phi - 180
        lats.append(lat)
        lons.append(lon)
    if axis is None:
        plt.figure()
        axis = plt.axes(projection=ccrs.Sinusoidal())
    axis.plot(lons, lats, "k.", transform=ccrs.PlateCarree())
    if with_labels:
        for s in range(0, len(event.hit_points)):
            axis.text(lons[s], lats[s], str(s + 1), transform=ccrs.Geodetic())
    axis.gridlines()
    axis.set_global()
    return axis


def plot_clusters_2d(
    event, clusters, ax=None, set_global=True, unordered=False, with_labels=False
):
    """Plot the clusters in 2D using a Sinusoidal projection"""
    point_matrix = event.point_matrix
    if ax is None:
        plt.figure()
        ax = plt.axes(projection=ccrs.Sinusoidal())

    for s in clusters.keys():
        if len(clusters[s]) == 0:
            continue
        lats = []
        lons = []
        xs = [point_matrix[i, 0] for i in clusters[s]]
        ys = [point_matrix[i, 1] for i in clusters[s]]
        zs = [point_matrix[i, 2] for i in clusters[s]]
        rs = [np.linalg.norm([x, y, z]) for (x, y, z) in zip(xs, ys, zs)]

        thetas = [math.acos(z / r) for (z, r) in zip(zs, rs)]
        phis = [math.atan2(y, x) for (x, y) in zip(xs, ys)]

        for theta, phi in zip(thetas, phis):
            lat = 180 / np.pi * theta - 90
            lon = 180 / np.pi * phi - 180
            lats.append(lat)
            lons.append(lon)
        lats = [(lat + 90) % 180 - 90 for lat in lats]
        lons = [(lon + 180) % 360 - 180 for lon in lons]
        if unordered:
            ax.plot(
                lons,
                lats,
                ".",
                transform=ccrs.Geodetic(),
                color=colors[s % len(colors)],
            )
        else:
            ax.plot([lons[0]], [lats[0]], "ko", transform=ccrs.Geodetic())
            ax.plot(
                lons,
                lats,
                ".-",
                transform=ccrs.Geodetic(),
                color=colors[s % len(colors)],
            )
            if with_labels:
                ax.text(lons[0], lats[0], str(s), transform=ccrs.Geodetic())

    if set_global:
        ax.set_global()
    ax.gridlines()
    plt.tight_layout()
    return ax


def plot_clusters_cross_sections(
    event: Event, clusters: Dict, detector: DetectorConfig = default_config
):
    """Plot the clusters in 2D cross-sections"""
    point_matrix = event.point_matrix
    plt.figure()
    plt.subplot(131)

    def plot_circles():
        # Plot inner and outer circle
        ax = plt.gca()
        circle1 = plt.Circle((0, 0), detector.inner_radius, color="k", fill=False)
        ax.add_patch(circle1)
        circle2 = plt.Circle((0, 0), detector.outer_radius, color="k", fill=False)
        ax.add_patch(circle2)

    plot_circles()
    for i, cluster in clusters.items():
        xs = point_matrix[cluster, 0]
        ys = point_matrix[cluster, 1]
        plt.plot(xs, ys, color=colors[i % len(colors)])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("y vs x")

    plt.subplot(132)
    plot_circles()
    for i, cluster in clusters.items():
        xs = point_matrix[cluster, 0]
        zs = point_matrix[cluster, 2]
        plt.plot(xs, zs, color=colors[i % len(colors)])
        plt.xlabel("x")
        plt.ylabel("z")
        plt.title("z vs x")

    plt.subplot(133)
    plot_circles()
    for i, cluster in clusters.items():
        ys = point_matrix[cluster, 1]
        zs = point_matrix[cluster, 2]
        plt.plot(ys, zs, color=colors[i % len(colors)])
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("z vs y")
    plt.tight_layout()


# %% 2D plots for flattened events
def plot_flat_event(
    event: Event,
    clusters: dict,
    include_origin: bool = True,
    include_outer_radius: bool = True,
    with_labels: bool = False,
    detector: DetectorConfig = default_config,
    randomize: bool = False,
    random_seed: int = None,
    **plot_kwargs,
):
    """
    # Plot a flat gamma-ray event with clusters of interactions.

    This function plots a gamma-ray event that has been flattened (the third
    dimension has been removed; sequential angles and distances are maintained).
    If the event is not already flattened, it flattens the event as well using
    the `flatten_event` function.

    ## Arguments
    `event` : Event
        Gamma-ray event containing interactions.
    `clusters` : dict
        Dictionary containing ordered clusters of gamma-ray interactions.
    `include_origin` : bool, optional
        Should the origin be plotted. Default is True.
    `include_outer_radius` : bool, optional
        Should the outer radius of the detector be plotted. Default is True.
    `with_labels` : bool, optional
        Should cluster id labels be added to the first interaction point location. Default is False.
    `detector` : DetectorConfig, optional
        The detector configuration to be used. Default is `default_config`.
    `**plot_kwargs` : dict, optional
        Additional keyword arguments to be passed to the `plt.plot` function.

    ## Returns
    None

    ## Examples
    >>> from greto import default_event, default_clusters
    >>> from greto.cluster_viz import plot_flat_event
    >>> import matplotlib.pyplot as plt
    >>> plot_flat_event(default_event, default_clusters)
    >>> plt.show()
    # This will plot a flat event with two clusters of interactions,
    # with origin and outer radius included.

    >>> plot_flat_event(default_event, default_clusters, include_origin=False, with_labels=True)
    >>> plt.show()
    # This will plot a flat event with two clusters of interactions,
    # without origin but with cluster id labels.

    >>> plot_flat_event(default_event, default_clusters, color='red', linestyle='--')
    >>> plt.show()
    # This will plot a flat event with two clusters of interactions
    # in red dashed lines.
    """
    if not event.flat:
        event = flatten_event(
            event,
            clusters,
            correct_air_gap=True,
            randomize=randomize,
            random_seed=random_seed,
        )
    ax = plt.gca()
    circle1 = plt.Circle((0, 0), detector.inner_radius, color="k", fill=False)
    ax.add_patch(circle1)
    if include_outer_radius:
        circle2 = plt.Circle((0, 0), detector.outer_radius, color="k", fill=False)
        ax.add_patch(circle2)

    if include_origin:
        for cluster_id, cluster in clusters.items():
            plt.plot(
                event.point_matrix[[0] + list(cluster), 0],
                event.point_matrix[[0] + list(cluster), 1],
                marker=".",
                **plot_kwargs,
            )
            plt.scatter(
                event.point_matrix[cluster[0], 0],
                event.point_matrix[cluster[0], 1],
                marker="o",
                color="black",
                s=30,
            )
            if with_labels:
                ax.text(
                    event.point_matrix[cluster[0], 0],
                    event.point_matrix[cluster[0], 1],
                    str(cluster_id),
                )
    else:
        for cluster_id, cluster in clusters.items():
            plt.plot(
                event.point_matrix[list(cluster), 0],
                event.point_matrix[list(cluster), 1],
                marker=".",
                **plot_kwargs,
            )
            plt.scatter(
                event.point_matrix[cluster[0], 0],
                event.point_matrix[cluster[0], 1],
                marker="o",
                color="black",
                s=30,
            )
            if with_labels:
                ax.text(
                    event.point_matrix[cluster[0], 0],
                    event.point_matrix[cluster[0], 1],
                    str(cluster_id),
                )
    ax.axis("equal")
    return ax


def plot_flat_event_px(
    event: Event,
    clusters: dict,
    include_origin: bool = True,
    include_outer_radius: bool = True,
    detector: DetectorConfig = default_config,
):
    """
    # Plot a flat gamma-ray event with clusters of interactions.

    This function plots a gamma-ray event that has been flattened (the third
    dimension has been removed; sequential angles and distances are maintained).
    If the event is not already flattened, it flattens the event as well using
    the `flatten_event` function.

    This version uses plotly to perform the plotting operation allowing
    information when hovering over data points.

    ## Arguments
    `event` : Event
        Gamma-ray event containing interactions.
    `clusters` : dict
        Dictionary containing ordered clusters of gamma-ray interactions.
    `include_origin` : bool, optional
        Should the origin be plotted. Default is True.
    `include_outer_radius` : bool, optional
        Should the outer radius of the detector be plotted. Default is True.
    `detector` : DetectorConfig, optional
        The detector configuration to be used. Default is `default_config`.

    ## Returns
    None

    ## Examples
    >>> from greto import default_event, default_clusters
    >>> from greto.cluster_viz import plot_flat_event_px
    >>> plot_flat_event_px(default_event, default_clusters)
    # This will plot a flat event with clusters of interactions, with origin
    # and outer radius included.

    >>> plot_flat_event_px(default_event, default_clusters,
        include_origin=False, include_outer_radius=False)
    # This will plot a flat event with clusters of interactions, without origin or outer radius.
    """
    if not event.flat:
        # print('Flattening event')
        # print(event.flat)
        event = flatten_event(event, clusters, correct_air_gap=True)
        # print(event.flat)
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    highlight_shape = "cross"  # Define the marker shape for the highlighted point
    data = []
    # dashed_data = []
    for cluster_id, cluster in clusters.items():
        if include_origin:
            dashed_df = pd.DataFrame(
                event.point_matrix[[0] + [cluster[0]]], columns=["x", "y", "z"]
            )
            dashed_df["colors"] = [colors[cluster_id % len(colors)]] * 2
            dashed_trace = go.Scatter(
                x=dashed_df["x"],
                y=dashed_df["y"],
                mode="lines",
                showlegend=False,
                line={"color": colors[cluster_id % len(colors)], "dash": "dash"},
                hoverinfo="skip",  # Disable hover for the dashed segment
                opacity=0.5,
            )
            data.append(dashed_trace)
        df = pd.DataFrame(event.point_matrix[list(cluster)], columns=["x", "y", "z"])
        df["id"] = cluster
        df["cluster_id"] = [cluster_id] * len(cluster)
        df["colors"] = [colors[cluster_id % len(colors)]] * len(cluster)
        df["energy"] = event.energy_matrix[list(cluster)]
        df["type"] = [event.points[i].interaction_type for i in cluster]
        df["cumu_energy"] = event.cumulative_energies(tuple(cluster))
        df["angle"] = list(event.theta_act_perm(tuple(cluster))) + [0]
        df["hover_info"] = list(
            zip(
                df["id"],
                df["cluster_id"],
                df["energy"],
                df["cumu_energy"],
                df["angle"],
                df["type"],
            )
        )

        scatter_trace = go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="lines+markers",
            name=f"Cluster {cluster_id}",
            line={"color": colors[cluster_id % len(colors)]},
            hovertemplate="<b>Cluster ID:</b> %{text[1]}<br><b>Int ID:</b> %{text[0]}<br>"
            + "<b>e:</b> %{text[2]:4.4f}<br><b>E:</b> %{text[3]:4.4f}<br>"
            + "<b>Type:</b> %{text[5]}<br>"
            + "<b>θ:</b> %{text[4]:4.4f}<br><b>X:</b> %{x}<br><b>Y:</b> %{y}<extra></extra>",
            text=df["hover_info"],  # Use the tuple of hover information as the text
        )

        # Highlight the first point in the cluster
        scatter_trace.marker.symbol = [
            highlight_shape if i == 0 else "circle" for i in range(len(cluster))
        ]
        data.append(scatter_trace)

    fig = go.Figure(data=data)
    shapes = [
        {
            "type": "circle",
            "x0": -detector.inner_radius,
            "y0": -detector.inner_radius,
            "x1": detector.inner_radius,
            "y1": detector.inner_radius,
            "xref": "x",
            "yref": "y",
            "fillcolor": "lightgray",
            "line_color": "black",
            "opacity": 0.5,
            "layer": "below",
        }
    ]
    if include_outer_radius:
        shapes.append(
            {
                "type": "circle",
                "x0": -detector.outer_radius,
                "y0": -detector.outer_radius,
                "x1": detector.outer_radius,
                "y1": detector.outer_radius,
                "xref": "x",
                "yref": "y",
                "line": {"color": "black", "dash": "dash"},
                "opacity": 0.5,
                "layer": "below",
            }
        )
    fig.update_layout(
        shapes=shapes,
        plot_bgcolor="white",  # Change the background color to light blue
        xaxis={"showgrid": False, "zeroline": False},  # Turn off the x-axis grid
        yaxis={"showgrid": False, "zeroline": False},  # Turn off the y-axis grid
        width=800,
        height=800,
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# %% 3D plots
def plot_clusters_3d(
    event: Event,
    clusters: Dict,
    include_origin: bool = False,
    detector: DetectorConfig = default_config,
):
    """Plot the clusters in 3D"""
    point_matrix = event.point_matrix
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for s in clusters.keys():
        if len(clusters[s]) > 0:
            xs = [point_matrix[i, 0] for i in clusters[s]]
            ys = [point_matrix[i, 1] for i in clusters[s]]
            zs = [point_matrix[i, 2] for i in clusters[s]]

            if include_origin:
                ax.plot(
                    [0, xs[0]], [0, ys[0]], [0, zs[0]], color=colors[s % len(colors)]
                )
            if len(clusters[s]) == 1:
                ax.scatter(xs, ys, zs, color=colors[s % len(colors)], marker=".")
            for i in range(1, len(xs)):
                ax.plot(
                    xs[i - 1 : i + 1],
                    ys[i - 1 : i + 1],
                    zs[i - 1 : i + 1],
                    color=colors[s % len(colors)],
                    marker=".",
                )

    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
    inner_x = detector.inner_radius * np.cos(u) * np.sin(v)
    inner_y = detector.inner_radius * np.sin(u) * np.sin(v)
    inner_z = detector.inner_radius * np.cos(v)
    ax.plot_surface(inner_x, inner_y, inner_z, alpha=0.2)

    outer_x = detector.outer_radius * np.cos(u) * np.sin(v)
    outer_y = detector.outer_radius * np.sin(u) * np.sin(v)
    outer_z = detector.outer_radius * np.cos(v)
    ax.plot_surface(outer_x, outer_y, outer_z, alpha=0.2)

    ax.set_xlim((-35, 35))
    ax.set_ylim((-35, 35))
    ax.set_zlim((-35, 35))


def plot_event_3d(
    event: Event,
    clusters: dict,
    include_origin: bool = True,
    include_inner_radius: bool = True,
    include_outer_radius: bool = False,
    detector: DetectorConfig = default_config,
):
    """Method for plotting flat events (the third dimension has been removed)"""
    highlight_shape = "cross"  # Define the marker shape for the highlighted point
    data = []
    for cluster_id, cluster in clusters.items():
        if include_origin:
            dashed_df = pd.DataFrame(
                event.point_matrix[[0] + [cluster[0]]], columns=["x", "y", "z"]
            )
            dashed_df["colors"] = [colors[cluster_id % len(colors)]] * 2
            dashed_trace = go.Scatter3d(
                x=dashed_df["x"],
                y=dashed_df["y"],
                z=dashed_df["z"],
                mode="lines",
                showlegend=False,
                line=dict(color=colors[cluster_id % len(colors)], dash="dash"),
                hoverinfo="skip",  # Disable hover for the dashed segment
                opacity=0.5,
            )
            data.append(dashed_trace)
        df = pd.DataFrame(event.point_matrix[list(cluster)], columns=["x", "y", "z"])
        df["id"] = cluster
        df["cluster_id"] = [cluster_id] * len(cluster)
        df["colors"] = [colors[cluster_id % len(colors)]] * len(cluster)
        df["energy"] = event.energy_matrix[list(cluster)]
        df["type"] = [event.points[i].interaction_type for i in cluster]
        df["cumu_energy"] = event.cumulative_energies(tuple(cluster))
        df["tot_energy"] = np.ones(df["cumu_energy"].shape) * df["cumu_energy"][0]
        df["angle"] = list(event.theta_act_perm(tuple(cluster))) + [0]
        df["hover_info"] = list(
            zip(
                df["id"],
                df["cluster_id"],
                df["energy"],
                df["cumu_energy"],
                df["angle"],
                df["tot_energy"],
                df["type"],
            )
        )  # Store all hover information in a tuple

        scatter_trace = go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="lines+markers",
            name=f"Cluster {cluster_id}",
            line={"color": colors[cluster_id % len(colors)]},
            hovertemplate="<b>Cluster:</b> %{text[1]}<br><b>Int:</b> %{text[0]}<br>"
            + "<b>e/E:</b> %{text[2]:4.4f}/%{text[5]:4.4f}<br><b>Ei:</b> %{text[3]:4.4f}<br>"
            + "<b>Type</b> %{text[6]}<br>"
            + "<b>θ:</b> %{text[4]:4.4f}<br><b>X:</b> %{x}<br><b>Y:</b> %{y}<br>"
            + "<b>Z:</b> %{z}<extra></extra>",
            text=df["hover_info"],  # Use the tuple of hover information as the text
            marker={"size": 4},
        )

        # Highlight the first point in the cluster
        scatter_trace.marker.symbol = [
            highlight_shape if i == 0 else "circle" for i in range(len(cluster))
        ]
        # TODO - add different markers for different interaction types
        data.append(scatter_trace)

    def sphere_mesh(x, y, z, radius, resolution=20):
        """Return the coordinates for plotting a sphere centered at (x,y,z)"""
        u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]
        X = radius * np.cos(u) * np.sin(v) + x
        Y = radius * np.sin(u) * np.sin(v) + y
        Z = radius * np.cos(v) + z
        return (X, Y, Z)

    if include_inner_radius or include_outer_radius:
        spheres = []
        if include_inner_radius:
            spheres.append((0, 0, 0, detector.inner_radius))
        if include_outer_radius:
            spheres.append((0, 0, 0, detector.outer_radius))
        spheres = np.array(spheres)
        # df = pd.DataFrame(spheres, columns=['x', 'y', 'z', 'r'])

        colorscale = [[0, "rgb(0.5,0.5,0.5)"], [1, "rgb(0.5,0.5,0.5)"]]

        for x, y, z, r in spheres:
            (x_surface, y_surface, z_surface) = sphere_mesh(x, y, z, r, 50)
            surface_color = np.zeros(shape=x_surface.shape)
            mesh_trace = go.Surface(
                x=x_surface,
                y=y_surface,
                z=z_surface,
                surfacecolor=surface_color,
                showscale=False,
                colorscale=colorscale,
                cmin=0,
                cmax=1,
                opacity=0.1,
                hoverinfo="skip",
                uirevision="surface",
            )
            data.append(mesh_trace)

    fig = go.Figure(data=data)
    fig.update_layout(width=800, height=800)
    # fig.show()
    return fig


def plot_points_3d(
    event: Event,  # point_matrix:np.ndarray = None,
    include_inner_radius: bool = True,
    include_outer_radius: bool = False,
    detector: DetectorConfig = default_config,
):
    """Method for plotting flat events (the third dimension has been removed)"""
    data = []
    # if point_matrix is None:
    #     hit_point_matrix = event.hit_point_matrix
    df = pd.DataFrame(event.hit_point_matrix, columns=["x", "y", "z"])
    # df['colors'] = [colors[cluster_id % len(colors)]] * len(cluster)
    df["id"] = np.arange(1, len(event.hit_point_matrix) + 1)
    df["energy"] = event.energy_matrix[1:]
    df["hover_info"] = list(
        zip(df["id"], df["energy"])
    )  # Store all hover information in a tuple

    scatter_trace = go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        mode="markers",
        hovertemplate="<b>Int:</b> %{text[0]}<br><b>Energy:</b> %{text[1]}<br>"
        + "<b>X:</b> %{x}<br><b>Y:</b> %{y}<br>"
        + "<b>Z:</b> %{z}<extra></extra>",
        text=df["hover_info"],  # Use the tuple of hover information as the text
        marker={"size": 4},
    )

    data.append(scatter_trace)

    def sphere_mesh(x, y, z, radius, resolution=20):
        """Return the coordinates for plotting a sphere centered at (x,y,z)"""
        u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]
        X = radius * np.cos(u) * np.sin(v) + x
        Y = radius * np.sin(u) * np.sin(v) + y
        Z = radius * np.cos(v) + z
        return (X, Y, Z)

    if include_inner_radius or include_outer_radius:
        spheres = []
        if include_inner_radius:
            spheres.append((0, 0, 0, detector.inner_radius))
        if include_outer_radius:
            spheres.append((0, 0, 0, detector.outer_radius))
        spheres = np.array(spheres)
        # df = pd.DataFrame(spheres, columns=['x', 'y', 'z', 'r'])

        colorscale = [[0, "rgb(0.5,0.5,0.5)"], [1, "rgb(0.5,0.5,0.5)"]]

        for x, y, z, r in spheres:
            (x_surface, y_surface, z_surface) = sphere_mesh(x, y, z, r, 50)
            surface_color = np.zeros(shape=x_surface.shape)
            mesh_trace = go.Surface(
                x=x_surface,
                y=y_surface,
                z=z_surface,
                surfacecolor=surface_color,
                showscale=False,
                colorscale=colorscale,
                cmin=0,
                cmax=1,
                opacity=0.1,
                hoverinfo="skip",
                uirevision="surface",
            )
            data.append(mesh_trace)

    fig = go.Figure(data=data)
    fig.update_layout(width=800, height=800)
    fig.show()


# %% Cluster sampling functions
def adjust_limits(ax):
    """Adjust the axis limits"""
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    dx = x2 - x1
    dy = y2 - y1

    if dx > dy:
        y1 -= (dx - dy) / 2
        y2 += (dx - dy) / 2
        ax.set_ylim(y1, y2)
    else:
        x1 -= (dy - dx) / 2
        x2 += (dy - dx) / 2
        ax.set_xlim(x1, x2)


def plot_bad_clustering_sample(
    list_of_events: List[Event],
    bad_idxs: List[int],
    list_of_pred_clusters: List[Dict],
    list_of_true_clusters: List[Dict],
):
    """
    # Plot a (potentially) bad clustering sample on the 2D projection

    Plots a sample of clusters that were marked as incorrect
    """
    plt.figure()
    for i in range(5):
        idx = bad_idxs[i]
        event = list_of_events[idx]
        ax = plt.subplot(5, 4, 4 * i + 1, projection=ccrs.PlateCarree())
        pred_clusters = list_of_pred_clusters[idx]

        plot_clusters_2d(event, pred_clusters, ax=ax, set_global=False)
        if i == 0:
            plt.title("Predicted (incorrect) Clustering")
        adjust_limits(ax)
        ax = plt.subplot(5, 4, 4 * i + 2, projection=ccrs.PlateCarree())
        true_clustering = list_of_true_clusters[idx]
        plot_clusters_2d(event, true_clustering, ax=ax, set_global=False)
        if i == 0:
            plt.title("True Clustering")
        adjust_limits(ax)

    for i in range(5):
        idx = bad_idxs[i + 5]
        event = list_of_events[idx]
        ax = plt.subplot(5, 4, 4 * i + 3, projection=ccrs.PlateCarree())
        pred_clusters = list_of_pred_clusters[idx]

        plot_clusters_2d(event, pred_clusters, ax=ax, set_global=False)
        if i == 0:
            plt.title("Predicted (incorrect) Clustering")
        adjust_limits(ax)
        ax = plt.subplot(5, 4, 4 * i + 4, projection=ccrs.PlateCarree())
        true_clustering = list_of_true_clusters[idx]
        plot_clusters_2d(event, true_clustering, ax=ax, set_global=False)
        if i == 0:
            plt.title("True Clustering")
        adjust_limits(ax)


def plot_right_wrong_histogram(events, tracks, true_tracks):
    """Plot a histogram of right and wrong energies"""
    mismatch_energies = cluster_utils.compute_mismatch_counts(
        events, tracks, true_tracks
    )
    energies = {
        "Right": mismatch_energies[0],
        "Wrong": sum(
            [mismatch_energies[i] for i in mismatch_energies if i > 0], start=[]
        ),
    }
    plt.figure()
    max_energy = max(max(v) for v in energies.values())
    sns.histplot(
        energies,
        bins=np.arange(0, max_energy + 0.004, 0.004),
        element="step",
        alpha=0.3,
    )
    plt.yscale("log")


def plot_energy_movement_histogram(
    events, tracks, true_tracks, observed_energies, **kwargs
):
    """Plot an energy movement histogram"""
    reclustered_energies = cluster_utils.compute_reclustered_energies(
        events, tracks, true_tracks, observed_energies
    )
    plt.figure()
    max_energy = max(max(v) for v in reclustered_energies.values())
    hist_kwargs = {
        "bins": np.arange(0, max_energy + 0.004, 0.004),
        "element": "step",
        "alpha": 0.3,
    }
    hist_kwargs.update(kwargs)

    sns.histplot(reclustered_energies, **hist_kwargs)
    plt.yscale("log")
    plt.title("Energy Movement after Clustering")


def resampled_cumulant(data, t, normalize=True):
    """
    Cumulant data can be plotted by simply sorting the data (horizontal)
    and plotting versus an regular index (vertical). This function resamples
    that data to get the cumulant as a function of a regular (or supplied)
    horizontal axis, i.e., makes the regular axis the horizontal.

    Args:
        data (iterable): The data to gather cumulant data from
        t (int or np.array): Specifies the points at which the cumulant is to be
            evaluated. If an int, create an evenly spaced array.
        normalize (boolean): Normalize the cumulant to be between zero and one
            or report the raw counts.
    Returns:
        out (np.ndarray): The cumulant
        t (np.ndarray): The points the cumulant was evaluated at
    """
    if isinstance(t, int):
        percentile = np.linspace(0, 1, t)
        t = np.quantile(data, percentile, method="inverted_cdf")
        if not normalize:
            percentile = percentile * len(data)
        # return (percentile, t)
    percentile = percentileofscore(data, t) / 100
    if not normalize:
        percentile = percentile * len(data)
    return (percentile, t)


def pt_eff(
    foms,
    energy_data: np.ndarray = None,
    peak_energies: list = None,
    peak_energy_tolerance: float = 1e-2,
    fom_range: np.ndarray = 100,
    peak_data_indicator: np.ndarray = None,
    return_fom_range: bool = False,
):
    """
    Get peak to total and efficiency at various FOM values
    - foms: the evaluated figure-of-merit (FOM) for each g-ray
    - peak_energies: the expected peak energies (e.g., [1.173, 1.333] for Co60)
    - peak_energy_tolerance: if energy is within this of a peak, assume it's in the peak
    - fom_range: range of FOM cutoff values to evaluate P/T and Eff at
    - peak_data_indicator: label if the data falls into a peak (pre-evaluated)

    Returns
    - peak_to_total: P/T ratio at the provided FOM cutoff values
    - efficiency: Efficiency at the provided FOM cutoff values
    """
    if peak_data_indicator is None and peak_energies is None:
        raise ValueError("Either peak energies or a peak indicator must be provided.")
    if peak_data_indicator is None:
        peak_data_indicator = detect_peaks(
            energy_data, peak_energies, peak_energy_tolerance
        )
    non_peak_indicator = np.logical_not(peak_data_indicator)
    quantity_peak, t = resampled_cumulant(
        np.nan_to_num(foms[peak_data_indicator]), t=fom_range, normalize=False
    )
    quantity_non_peak, t = resampled_cumulant(
        np.nan_to_num(foms[non_peak_indicator]), t=t, normalize=False
    )
    peak_to_total = quantity_peak / (np.maximum(quantity_peak + quantity_non_peak, 1))
    efficiency = (quantity_peak + quantity_non_peak) / (foms.shape[0])
    if return_fom_range:
        return peak_to_total, efficiency, t
    return peak_to_total, efficiency


def detect_peaks(
    energies: np.ndarray, peak_energies: list[float], tol: float = 1e-2
) -> np.ndarray:
    """Detect if energies are within some tolerance of some provided peak energies

    - energies: array of energy values
    - peak_energies: list of peak energy values
    - tol: tolerance where energy would be considered in the peak

    Returns
    - in_peak: boolean array indicating if the energy is in the peak or not
    """
    in_peak = np.zeros(energies.shape, dtype=bool)
    for p_energy in peak_energies:
        in_peak[np.abs(energies - p_energy) < tol] = True
    return in_peak
