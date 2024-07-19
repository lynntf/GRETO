"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Methods for executing the tracking of gamma-rays.
"""

import pickle as pkl
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process, Queue, cpu_count
from typing import BinaryIO, ByteString, Dict, List, Tuple

import multiprocess as mp
import numpy as np
from tqdm import tqdm

from greto.cluster_tools import (
    cluster_linkage,
    cone_cluster_linkage,
    join_events,
    pack_interactions,
    remove_zero_energy_interactions,
    split_event,
)
from greto.detector_config_class import default_config
from greto.event_class import Event
from greto.file_io import (
    load_options,
    mode1_data,
    mode1_extended_data,
    mode1_loader,
    mode2_loader,
    read_agata_simulated_data,
    tracked_generator,
)
from greto.fom_tools import (
    cluster_FOM,
    cluster_model_FOM,
    semi_greedy,
    semi_greedy_batch,
    semi_greedy_batch_clusters,
    semi_greedy_clusters,
)
from greto.models import load_order_FOM_model, load_suppression_FOM_model, load_models
from greto.physics import inv_doppler
from greto.utils import get_file_size


def track_files(mode2file: BinaryIO, output_file: BinaryIO, options: Dict):
    """
    Take in the mode2 file to track, the mode1 file to write, and tracking
    options
    """
    order_FOM_kwargs = options.get("order_FOM_kwargs", {})
    secondary_order_FOM_kwargs = options.get("secondary_order_FOM_kwargs", {})
    eval_FOM_kwargs = options.get("eval_FOM_kwargs", {})
    cluster_kwargs = options.get("cluster_kwargs", {})

    print(f"Detector is set to {options.get('DETECTOR', 'defaulting to GRETINA')}")
    default_config.set_detector(options.get("DETECTOR", "GRETINA"))

    if options.get("SAVE_INTERMEDIATE", False):
        print("Tracking will save an intermediate tracked format instead of mode1")
    elif options.get("SAVE_EXTENDED_MODE1", False):
        print("Tracking will save an extended mode1 data format with all interactions")

    order_model = None
    if order_FOM_kwargs.get("fom_method") == "model":
        if order_FOM_kwargs.get("model_filename") is not None:
            order_model = load_order_FOM_model(order_FOM_kwargs.get("model_filename"))
        else:
            raise ValueError("Provide model filename for ordering")
    order_FOM_kwargs["model"] = order_model

    secondary_order_model = None
    if secondary_order_FOM_kwargs.get("fom_method") == "model":
        if secondary_order_FOM_kwargs.get("model_filename") is not None:
            secondary_order_model = load_order_FOM_model(
                secondary_order_FOM_kwargs.get("model_filename")
            )
        else:
            raise ValueError("Provide model filename for secondary ordering")
    secondary_order_FOM_kwargs["model"] = secondary_order_model

    eval_model = None
    if eval_FOM_kwargs.get("fom_method") == "model":
        if eval_FOM_kwargs.get("model_filename") is not None:
            eval_model = load_suppression_FOM_model(
                eval_FOM_kwargs.get("model_filename")
            )
        else:
            raise ValueError("Provide model filename for evaluation")
    eval_FOM_kwargs["model"] = eval_model

    GEB_EVENT_SIZE = 480
    filesize = get_file_size(mode2file)
    n_data_points = filesize // GEB_EVENT_SIZE
    if options["PARTIAL_TRACK"]:
        final_position = options["PARTIAL_TRACK"]
    else:
        final_position = filesize

    start = datetime.now()

    print(f'[{start.strftime("%H:%M:%S")}] Start')

    chunk_size = options["NUM_PROCESSES"] * 100
    # chunk_size = options["NUM_PROCESSES"] * 1
    chunks = max(n_data_points // chunk_size, 1)
    chunk = 0
    num_events = 1
    previous_position = 1

    # if options["VERBOSITY"] >= 3:
    #     progress_bar = tqdm(total=final_position, unit="bytes", unit_scale=True)

    if options["NUM_PROCESSES"] > 1:
        with mp.Pool(options["NUM_PROCESSES"]) as pool:  # pylint: disable=E1102
            if not options["READ_MODE1"]:
                loader = mode2_loader(
                    mode2file,
                    time_gap=options["COINCIDENCE_TIME_GAP"],
                    monitor_progress=False,
                    global_coords=options["GLOBAL_COORDS"],
                    print_formatted=options["PRINT_FORMATTED"],
                )
            else:
                loader = mode1_loader(
                    mode2file, print_formatted=options["PRINT_FORMATTED"]
                )
            tracking_processes = []
            coincidence_events = []
            for num_events, coincidence_event in enumerate(loader):
                if coincidence_event is not None:
                    coincidence_event = remove_zero_energy_interactions(
                        coincidence_event
                    )
                    coincidence_event = pack_interactions(
                        coincidence_event, packing_distance=1e-8
                    )
                    coincidence_events.append(coincidence_event)
                    tracking_processes.append(
                        pool.apply_async(
                            track_event,
                            args=(
                                coincidence_event,
                                options["MONSTER_SIZE"],
                                options["SECONDARY_ORDER"],
                                eval_FOM_kwargs,
                                options["MAX_HIT_POINTS"],
                                cluster_kwargs,
                                order_FOM_kwargs,
                                secondary_order_FOM_kwargs,
                                options["SAVE_INTERMEDIATE"],
                                options["SAVE_EXTENDED_MODE1"],
                                options["TRACK_MOLY_PEAK"],
                                options["RECORD_UNTRACKED"],
                                options["SUPPRESS_BAD_PAD"],
                            ),
                        )
                    )
                position = mode2file.tell()
                if (
                    position // (chunk_size * GEB_EVENT_SIZE) > chunk
                    or position >= final_position
                ):
                    chunk += 1
                    elapsed_time = datetime.now() - start
                    average_time = (elapsed_time) // (max(1, chunk - 1))
                    if options["VERBOSITY"] >= 2:
                        print(
                            f'[{datetime.now().strftime("%H:%M:%S")} || {elapsed_time}]'
                            + f"  Processing chunk {chunk} of {chunks + 1}"
                        )
                        print(
                            f"           Progress {previous_position} of {final_position} || "
                            + f"{previous_position/final_position*100:2.2f}%"
                        )
                        print(f"           Events {num_events}")
                        print(f"  Average time per chunk:   {average_time}")
                        print(
                            f"  Average time per event:   {elapsed_time/max(1, num_events)}"
                        )
                        print(
                            "  Est. time remaining:      "
                            + f"{elapsed_time*(final_position/previous_position - 1)}"
                        )
                    for j, tracking_outputs in enumerate(tracking_processes):
                        try:
                            outputs = tracking_outputs.get(
                                timeout=options["TIMEOUT_SECONDS"]
                            )
                        except mp.TimeoutError:  # pylint: disable=E1101
                            print(f"Timeout occurred for process {j}")
                            print(coincidence_events[j])
                        except ValueError as e:
                            print(f"Found a bad event at process {j}: {e}")
                        else:
                            output_file.write(outputs)
                    tracking_processes = []
                    # if options["VERBOSITY"] >= 3:
                    #     progress_bar.update(position - previous_position)
                    previous_position = position
                    if position >= final_position:
                        break
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Completed')
        print(f"[Total time : {datetime.now() - start}].")
        print(
            f"[Average event processing time : {(datetime.now() - start)/max(1,num_events)}]."
        )
    elif options["NUM_PROCESSES"] == 1:
        print("Using single process")
        if not options["READ_MODE1"]:
            loader = mode2_loader(
                mode2file,
                time_gap=options["COINCIDENCE_TIME_GAP"],
                monitor_progress=False,
                global_coords=options["GLOBAL_COORDS"],
                print_formatted=options["PRINT_FORMATTED"],
            )
        else:
            loader = mode1_loader(mode2file, print_formatted=options["PRINT_FORMATTED"])
        outputs = []
        for num_events, coincidence_event in enumerate(loader):
            if coincidence_event is not None:
                coincidence_event = remove_zero_energy_interactions(coincidence_event)
                coincidence_event = pack_interactions(
                    coincidence_event, packing_distance=0.0
                )
                try:
                    outputs.append(
                        track_event(
                            coincidence_event,
                            options["MONSTER_SIZE"],
                            options["SECONDARY_ORDER"],
                            eval_FOM_kwargs,
                            options["MAX_HIT_POINTS"],
                            cluster_kwargs,
                            order_FOM_kwargs,
                            secondary_order_FOM_kwargs,
                            options["SAVE_INTERMEDIATE"],
                            options["SAVE_EXTENDED_MODE1"],
                            options["TRACK_MOLY_PEAK"],
                            options["RECORD_UNTRACKED"],
                            options["SUPPRESS_BAD_PAD"],
                        )
                    )
                except ValueError as ex:
                    print(f"Some kind of error: {ex}")
            if options["VERBOSITY"] >= 4:
                print(coincidence_event)
            position = mode2file.tell()
            if (
                position // (chunk_size * GEB_EVENT_SIZE) > chunk
                or position >= final_position
            ):
                chunk += 1
                elapsed_time = datetime.now() - start
                average_time = (elapsed_time) // (max(1, chunk - 1))
                if options["VERBOSITY"] >= 2:
                    print(
                        f'[{datetime.now().strftime("%H:%M:%S")} || {elapsed_time}]'
                        + f"  Processing chunk {chunk} of {chunks + 1}"
                    )
                    print(
                        f"           Progress {previous_position} of {final_position} || "
                        + f"{previous_position/final_position*100:2.2f}%"
                    )
                    print(f"           Events {num_events}")
                    print(f"  Average time per chunk:   {average_time}")
                    print(
                        f"  Average time per event:   {elapsed_time/max(1, num_events)}"
                    )
                    print(
                        "  Est. time remaining:      "
                        + f"{elapsed_time*(final_position/previous_position - 1)}"
                    )
                for output in outputs:
                    output_file.write(output)
                outputs = []
                # if options["VERBOSITY"] >= 3:
                #     progress_bar.update(position - previous_position)
                previous_position = position
                if position >= final_position:
                    break
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Completed')
        print(f"[Total time : {datetime.now() - start}].")
        print(
            f"[Average event processing time : {(datetime.now() - start)/max(1,num_events)}]."
        )


def track_simulated(events: List[Event], output_file: BinaryIO, options: Dict):
    """
    Take in the mode2 file to track, the mode1 file to write, and tracking
    options
    """
    order_FOM_kwargs = options["order_FOM_kwargs"]
    secondary_order_FOM_kwargs = options["secondary_order_FOM_kwargs"]
    eval_FOM_kwargs = options["eval_FOM_kwargs"]
    cluster_kwargs = options["cluster_kwargs"]

    print(f"Detector is set to {options['DETECTOR']}")
    default_config.set_detector(options["DETECTOR"])

    if options["SAVE_INTERMEDIATE"]:
        print("Tracking will save an intermediate tracked format instead of mode1")
    elif options["SAVE_EXTENDED_MODE1"]:
        print("Tracking will save an extended mode1 data format with all interactions")

    n_data_points = len(events)

    start = datetime.now()

    print(f'[{start.strftime("%H:%M:%S")}] Start')

    chunk_size = options["NUM_PROCESSES"] * 100
    chunks = max(n_data_points // chunk_size, 1)
    chunk = 0
    num_events = 1

    # if options["VERBOSITY"] >= 3:
    #     progress_bar = tqdm(total=n_data_points, unit="Events", unit_scale=True)
    with mp.Pool(options["NUM_PROCESSES"]) as pool:  # pylint: disable=E1102
        tracking_processes = []
        for num_events, coincidence_event in enumerate(events):
            if coincidence_event is not None:
                coincidence_event = remove_zero_energy_interactions(coincidence_event)
                tracking_processes.append(
                    pool.apply_async(
                        track_event,
                        args=(
                            coincidence_event,
                            options["MONSTER_SIZE"],
                            options["SECONDARY_ORDER"],
                            eval_FOM_kwargs,
                            options["MAX_HIT_POINTS"],
                            cluster_kwargs,
                            order_FOM_kwargs,
                            secondary_order_FOM_kwargs,
                            options["SAVE_INTERMEDIATE"],
                            options["SAVE_EXTENDED_MODE1"],
                            options["TRACK_MOLY_PEAK"],
                        ),
                    )
                )
            if num_events // (chunk_size) > chunk or num_events == n_data_points - 1:
                chunk += 1
                elapsed_time = datetime.now() - start
                average_time = (elapsed_time) // (max(1, chunk - 1))
                if options["VERBOSITY"] >= 2:
                    print(
                        f'[{datetime.now().strftime("%H:%M:%S")} || {elapsed_time}]'
                        + f"  Processing chunk {chunk} of {chunks + 1}"
                    )
                    print(
                        f"           Progress {num_events} of {n_data_points} || "
                        + f"{num_events/n_data_points*100:2.2f}%"
                    )
                    print(f"           Events {num_events}")
                    print(f"  Average time per chunk:   {average_time}")
                    print(
                        f"  Average time per event:   {elapsed_time/max(1, num_events)}"
                    )
                    print(
                        "  Est. time remaining:      "
                        + f"{elapsed_time*(n_data_points/num_events - 1)}"
                    )
                for j, tracking_outputs in enumerate(tracking_processes):
                    try:
                        outputs = tracking_outputs.get(
                            timeout=options["TIMEOUT_SECONDS"]
                        )
                    except mp.TimeoutError:  # pylint: disable=E1101
                        print(f"Timeout occurred for process {j}")
                    except ValueError:
                        print(f"Found a bad event at process {j}")
                    else:
                        output_file.write(outputs)
                tracking_processes = []
                # if options["VERBOSITY"] >= 3:
                #     progress_bar.update(chunk_size)
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Completed')
    print(f"[Total time : {datetime.now() - start}].")
    print(
        f"[Average event processing time : {(datetime.now() - start)/max(1,num_events)}]."
    )


def track_simulated_serial(events: List[Event], output_file: BinaryIO, options: Dict):
    """
    Take in the mode2 file to track, the mode1 file to write, and tracking
    options
    """
    order_FOM_kwargs = options["order_FOM_kwargs"]
    secondary_order_FOM_kwargs = options["secondary_order_FOM_kwargs"]
    eval_FOM_kwargs = options["eval_FOM_kwargs"]
    cluster_kwargs = options["cluster_kwargs"]

    print(f"Detector is set to {options['DETECTOR']}")
    default_config.set_detector(options["DETECTOR"])

    if options["SAVE_INTERMEDIATE"]:
        print("Tracking will save an intermediate tracked format instead of mode1")
    elif options["SAVE_EXTENDED_MODE1"]:
        print("Tracking will save an extended mode1 data format with all interactions")

    start = datetime.now()

    print(f'[{start.strftime("%H:%M:%S")}] Start')

    for num_events, coincidence_event in enumerate(tqdm(events)):
        if coincidence_event is not None:
            coincidence_event = remove_zero_energy_interactions(coincidence_event)
            try:
                outputs = track_event(
                    coincidence_event,
                    options["MONSTER_SIZE"],
                    options["SECONDARY_ORDER"],
                    eval_FOM_kwargs,
                    options["MAX_HIT_POINTS"],
                    cluster_kwargs,
                    order_FOM_kwargs,
                    secondary_order_FOM_kwargs,
                    options["SAVE_INTERMEDIATE"],
                    options["SAVE_EXTENDED_MODE1"],
                )
            except ValueError as e:
                print(f"Found a bad event at id {num_events}")
                print(e)
            else:
                output_file.write(outputs)
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Completed')
    print(f"[Total time : {datetime.now() - start}].")
    print(
        f"[Average event processing time : {(datetime.now() - start)/max(1,len(events))}]."
    )


def cone_cluster(
    event: Event,
    MAX_HIT_POINTS: int = 100,
    cluster_kwargs: Dict = None,
) -> Dict:
    """Perform the cone clustering"""
    if cluster_kwargs is None:
        cluster_kwargs = {}

    if len(event.hit_points) == 0 or len(event.hit_points) > MAX_HIT_POINTS:
        print("Found a bad event:")
        print(event)
        raise ValueError(f" Event too large or empty: {len(event.hit_points)}")

    return cone_cluster_linkage(event, **cluster_kwargs)
    # return cluster_linkage(event, **cluster_kwargs)


def order_clusters(
    event: Event,
    pred_clusters: Dict,
    order_FOM_kwargs: Dict = None,
    split_event_by_cluster: bool = True,
    cluster_track_indicator: Dict = None,
) -> Tuple[Event, Dict]:
    """
    Perform clustering and then ordering, returning the ordered clusters and
    matching event

    - event: gamma-ray event
    - order_FOM_kwargs: keyword arguments for the ordering FOM. See
      fom_tools.FOM for details
    - split_event_by_cluster: split the event into smaller events (by cluster)
      before ordering, may save some computation

    Returns
    - event: should be roughly equivalent to the original event, but if the
    event was split and rejoined, the point indices may have changed
    - pred_clusters: the predicted clusters (possibly altered by splitting and
    joining)
    """
    if order_FOM_kwargs is None:
        order_FOM_kwargs = {}
    if cluster_track_indicator is None:
        cluster_track_indicator = {i: True for i in pred_clusters}

    if split_event_by_cluster:
        split_events, split_clusters = split_event(event, pred_clusters)

        # Order each split event/cluster
        ordered_clusters = []
        for ev, clusters in zip(split_events, split_clusters):
            clu = list(clusters.values())[0]
            cluster_id = list(clusters.keys())[0]
            if order_FOM_kwargs.get("model", None) is not None:
                ordered_clusters.append(
                    {
                        cluster_id: semi_greedy_batch(
                            ev,
                            clu,
                            track_indicator=cluster_track_indicator[cluster_id],
                            **order_FOM_kwargs,
                        )
                    }
                )
            else:
                ordered_clusters.append(
                    {
                        cluster_id: semi_greedy(
                            ev,
                            clu,
                            track_indicator=cluster_track_indicator[cluster_id],
                            **order_FOM_kwargs,
                        )
                    }
                )

        # Recombine the split event into a single event
        event, pred_clusters = join_events(split_events, ordered_clusters)
    else:
        if order_FOM_kwargs.get("model", None) is not None:
            pred_clusters = semi_greedy_batch_clusters(
                event,
                pred_clusters,
                cluster_track_indicator=cluster_track_indicator,
                **order_FOM_kwargs,
            )
        else:
            pred_clusters = semi_greedy_clusters(
                event,
                pred_clusters,
                cluster_track_indicator=cluster_track_indicator,
                **order_FOM_kwargs,
            )

    return event, pred_clusters


def solve_clusters(
    event: Event,
    MAX_HIT_POINTS: int = 100,
    cluster_kwargs: Dict = None,
    order_FOM_kwargs: Dict = None,
    split_event_by_cluster: bool = True,
) -> Tuple[Event, Dict]:
    """
    Perform clustering and then ordering, returning the ordered clusters and
    matching event

    - event: gamma-ray event
    - MAX_HIT_POINTS: the maximum number of hit points allowed in an event
    - cluster_kwargs: keyword arguments for clustering. See
      cluster_utils.cluster_linkage for details
    - order_FOM_kwargs: keyword arguments for the ordering FOM. See
      fom_tools.FOM for details
    - split_event_by_cluster: split the event into smaller events (by cluster)
      before ordering, may save some computation

    Returns
    - event: should be roughly equivalent to the original event, but if the
    event was split and rejoined, the point indices may have changed
    - pred_clusters: the predicted clusters (clustered according to
    cluster_kwargs and ordered according to order_FOM_kwargs)
    """
    if cluster_kwargs is None:
        cluster_kwargs = {}
    if order_FOM_kwargs is None:
        order_FOM_kwargs = {}

    if len(event.hit_points) == 0 or len(event.hit_points) > MAX_HIT_POINTS:
        print("Found a bad event:")
        print(event)
        raise ValueError(f" Event too large or empty: {len(event.hit_points)}")

    # Split the event into individual clusters
    pred_clusters = cluster_linkage(event, **cluster_kwargs)
    if split_event_by_cluster:
        split_events, split_clusters = split_event(event, pred_clusters)

        # Order each split event/cluster
        ordered_clusters = []
        for ev, clusters in zip(split_events, split_clusters):
            clu = list(clusters.values())[0]
            if order_FOM_kwargs.get("model", None) is not None:
                ordered_clusters.append(
                    {1: semi_greedy_batch(ev, clu, batch_size=1000, **order_FOM_kwargs)}
                )
            else:
                ordered_clusters.append({1: semi_greedy(ev, clu, **order_FOM_kwargs)})

        # Recombine the split event into a single event
        event, pred_clusters = join_events(split_events, ordered_clusters)
    else:
        if order_FOM_kwargs.get("model", None) is not None:
            pred_clusters = semi_greedy_batch_clusters(
                event, pred_clusters, **order_FOM_kwargs
            )
        else:
            pred_clusters = semi_greedy_clusters(
                event, pred_clusters, **order_FOM_kwargs
            )

    return event, pred_clusters


def solve_clusters_secondary_fom(
    event: Event,
    MAX_HIT_POINTS: int = 100,
    cluster_kwargs: Dict = None,
    secondary_order_FOM_kwargs: Dict = None,
    split_event_by_cluster: bool = True,
) -> Dict:
    """
    Perform clustering and tracking in order to return a FOM as if that tracking
    had been done.
    """
    if cluster_kwargs is None:
        cluster_kwargs = {}
    if secondary_order_FOM_kwargs is None:
        secondary_order_FOM_kwargs = {}

    if len(event.hit_points) == 0 or len(event.hit_points) > MAX_HIT_POINTS:
        print("Found a bad event:")
        print(event)
        raise ValueError(f" Event too large or empty: {len(event.hit_points)}")

    pred_clusters = cluster_linkage(event, **cluster_kwargs)
    if split_event_by_cluster:
        split_events, split_clusters = split_event(event, pred_clusters)
        ordered_clusters = []
        for ev, clusters in zip(split_events, split_clusters):
            clu = list(clusters.values())[0]
            if secondary_order_FOM_kwargs.get("model", None) is not None:
                ordered_clusters.append(
                    {1: semi_greedy_batch(ev, clu, **secondary_order_FOM_kwargs)}
                )
            else:
                ordered_clusters.append(
                    {1: semi_greedy(ev, clu, **secondary_order_FOM_kwargs)}
                )
        event, pred_clusters = join_events(split_events, ordered_clusters)
    else:
        if secondary_order_FOM_kwargs.get("model", None) is not None:
            pred_clusters = semi_greedy_batch_clusters(
                event, pred_clusters, **secondary_order_FOM_kwargs
            )
        else:
            pred_clusters = semi_greedy_clusters(
                event, pred_clusters, **secondary_order_FOM_kwargs
            )

    if secondary_order_FOM_kwargs.get("model", None) is not None:
        return cluster_model_FOM(
            event,
            pred_clusters,
            secondary_order_FOM_kwargs.get("model"),
            **secondary_order_FOM_kwargs,
        )
    try:
        out = cluster_FOM(event, pred_clusters, **secondary_order_FOM_kwargs)
    except Exception as ex:
        print(event, pred_clusters)
        raise ex
    return out


def moly_peak_check(
    event: Event,
    clusters: Dict,
    beam_direction: np.ndarray = np.array([0.0, 0.0, 1.0]),
    beta: float = 0.0845,
    peak_energy_MeV: float = 2.066,
    threshold_MeV: float = 0.06,
):
    """
    Check if a cluster can fall within a small region of a Molybdenum peak that
    we care about for checking ordering
    """
    energy_sums = {
        s: sum(event.points[i].e for i in clu) for s, clu in clusters.items()
    }
    check = {s: False for s in clusters}
    for s, clu in clusters.items():
        for index in clu:
            cos_theta = (
                np.dot(beam_direction, event.points[index].x) / event.points[index].r
            )
            if (
                np.abs(inv_doppler(beta, cos_theta) * energy_sums[s] - peak_energy_MeV)
                < threshold_MeV
            ):
                check[s] = True
                break
    return check


def track_event(
    event: Event,
    MONSTER_SIZE: int = 8,
    SECONDARY_ORDER: bool = False,
    eval_FOM_kwargs: Dict = None,
    MAX_HIT_POINTS: int = 100,
    cluster_kwargs: Dict = None,
    order_FOM_kwargs: Dict = None,
    secondary_order_FOM_kwargs: Dict = None,
    SAVE_INTERMEDIATE: bool = False,
    SAVE_EXTENDED_MODE1: bool = False,
    TRACK_MOLY_PEAK: bool = False,
    RECORD_UNTRACKED: bool = True,
    SUPPRESS_BAD_PAD: bool = False,
    regurgitate: bool = False,
    return_stats: bool = False,
    **kwargs,
) -> ByteString:
    """
    Track a single event by clustering and then ordering.

    Args:
    - event: g-ray event object
    - monster_size: maximum number of interactions that will be tracked per cluster
    - SECONDARY_ORDER: option that tracks a second time for a FOM (first track is order)
    - eval_FOM_kwargs: keyword arguments for FOM recorded in output
    - MAX_HIT_POINTS: maximum number of interaction per event (more than this will ignore the event)
    - cluster_kwargs: keyword arguments for clustering (cone clustering)
    - order_FOM_kwargs: keyword arguments for FOM used to order interactions
    - secondary_order_FOM_kwargs: keyword arguments for FOM used to get a secondary order (used for FOM, not order)
    - SAVE_INTERMEDIATE: DO NOT USE, saves a pickled version of the event to file instead of mode1(x) output
    - SAVE_EXTENDED_MODE1: saves a mode1x file instead of a mode1 file
    - TRACK_MOLY_PEAK: used to track just around a Molybdenum peak to check ordering
    - RECORD_UNTRACKED: record untracked data to the mode1(x) output
    - SUPPRESS_BAD_PAD: don't track any clusters that have a bad pad interaction in them
    - regurgitate: don't do any tracking at all, just regurgitate the data that would have been tracked

    Returns:
    - BytesString with mode1(x) bytes to write to file
    """
    # print("MONSTER_SIZE", MONSTER_SIZE)
    # print("SECONDARY_ORDER", SECONDARY_ORDER)
    # print("eval_FOM_kwargs", eval_FOM_kwargs)
    # print("MAX_HIT_POINTS", MAX_HIT_POINTS)
    # print("cluster_kwargs", cluster_kwargs)
    # print("order_FOM_kwargs", order_FOM_kwargs)
    # print("secondary_order_FOM_kwargs", secondary_order_FOM_kwargs)
    # print("SAVE_INTERMEDIATE", SAVE_INTERMEDIATE)
    # print("SAVE_EXTENDED_MODE1", SAVE_EXTENDED_MODE1)
    # print("TRACK_MOLY_PEAK", TRACK_MOLY_PEAK)
    # print("RECORD_UNTRACKED", RECORD_UNTRACKED)
    # print("SUPPRESS_BAD_PAD", SUPPRESS_BAD_PAD)
    # print("regurgitate", regurgitate)
    stats = defaultdict(int)
    for point in event.hit_points:
        stats["pad " + str(point.pad)] += 1
        stats["crystal " + str(point.crystal_no)] += 1

    if eval_FOM_kwargs is None:
        eval_FOM_kwargs = {}
    if cluster_kwargs is None:
        cluster_kwargs = {}
    if order_FOM_kwargs is None:
        order_FOM_kwargs = {}
    if secondary_order_FOM_kwargs is None:
        secondary_order_FOM_kwargs = {}

    clusters = cone_cluster(event, MAX_HIT_POINTS, cluster_kwargs)

    stats["clusters"] += len(clusters)

    if regurgitate:
        cluster_track_indicator = {s: False for s in clusters}
        regurgitate_indicator = moly_peak_check(event, clusters)
    else:
        cluster_track_indicator = {s: True for s in clusters}

    if TRACK_MOLY_PEAK:
        # indicator = {s: False for s in clusters}
        moly_indicator = moly_peak_check(event, clusters)
        cluster_track_indicator = {
            s: moly_indicator[s] and len(cluster) <= MONSTER_SIZE
            for s, cluster in clusters.items()
        }
    if SUPPRESS_BAD_PAD:
        indicator = {
            s: not any([event.points[i].pad > 0 for i in cluster])
            for s, cluster in clusters.items()
        }
        for i, cluster in clusters.items():
            for index in cluster:
                if event.points[index].pad > 0:
                    print(f"Found a bad pad, skipping:{event}")
        cluster_track_indicator = {
            s: indicator[s] and cluster_track_indicator[s] for s in clusters
        }
    gr_event, clusters = order_clusters(
        event,
        clusters,
        order_FOM_kwargs,
        cluster_track_indicator=cluster_track_indicator,
    )

    if SAVE_INTERMEDIATE:
        if not return_stats:
            return pkl.dumps((gr_event.coincidence, clusters))
        return pkl.dumps((gr_event.coincidence, clusters)), stats

    if SECONDARY_ORDER:
        # foms = solve_clusters_secondary_fom(
        #     event, MAX_HIT_POINTS, cluster_kwargs, secondary_order_FOM_kwargs
        # )
        _, secondary_clusters = order_clusters(
            event,
            clusters,
            secondary_order_FOM_kwargs,
            cluster_track_indicator=cluster_track_indicator,
        )
        foms = cluster_FOM(gr_event, secondary_clusters, **eval_FOM_kwargs)
        # foms = cluster_FOM(gr_event, secondary_clusters, **secondary_order_FOM_kwargs)
    else:
        foms = cluster_FOM(gr_event, clusters, **eval_FOM_kwargs)

    if not regurgitate:
        regurgitate_indicator = cluster_track_indicator
    if SAVE_EXTENDED_MODE1:
        output_bytes = mode1_extended_data(
            gr_event,
            clusters,
            foms=foms,
            monster_size=MONSTER_SIZE,
            tracked_dict=regurgitate_indicator,
            # tracked_dict=cluster_track_indicator,
            RECORD_UNTRACKED=RECORD_UNTRACKED,
            **eval_FOM_kwargs,
        )
    else:
        output_bytes = mode1_data(
            gr_event,
            clusters,
            foms=foms,
            monster_size=MONSTER_SIZE,
            tracked_dict=regurgitate_indicator,
            # tracked_dict=cluster_track_indicator,
            RECORD_UNTRACKED=RECORD_UNTRACKED,
            **eval_FOM_kwargs,
        )
    if not return_stats:
        return output_bytes
    return output_bytes, stats


def load_and_track_files(
    mode2_filename: str, mode1_filename: str, options_filename: str
):
    """
    Load files for reading and writing and then track them using the specified
    options
    """
    options = load_options(options_filename)
    with open(mode2_filename, "rb") as mode2file:
        print(f"Beginning tacking of Mode2 file {mode2_filename}.")
        with open(mode1_filename, "wb") as mode1file:
            print(f"Saving Mode1 data to {mode1_filename}.")
            track_files(mode2file, mode1file, options)


def load_and_track_simulated(
    simulated_data_filename: str, mode1_filename: str, options_filename: str
):
    """
    Load files for reading and writing and then track them using the specified
    options
    """
    options = load_options(options_filename)
    events, _ = read_agata_simulated_data(simulated_data_filename)
    print(f"Beginning tacking of simulated data file {simulated_data_filename}.")
    with open(mode1_filename, "wb") as mode1file:
        print(f"Saving Mode1 data to {mode1_filename}.")
        track_simulated(events, mode1file, options)


def evaluate_mode1x(
    tracked_data_filename: str, output_filename: str, options_filename: str
):
    """
    Load previously tracked data and apply a validation FOM for suppression
    using the specified options

    TODO - fix the generator, it's using pickle...
    """
    options = load_options(options_filename)
    print(f"Loading the previously tracked data from {tracked_data_filename}")
    with open(tracked_data_filename, "rb") as tracked_file:
        print(f"Saving evaluation data to output file {output_filename}")
        with open(output_filename, "wb") as mode1file:
            print(f"Using evaluation FOM: {options.eval_FOM_kwargs}")
            for event, clusters in tracked_generator(tracked_file):
                mode1file.write(mode1_data(event, clusters, **options.eval_FOM_kwargs))


def process_event(event: Event, **options):
    """
    Track each event

    Args
    ----
    event: Event. the g-ray event to track
    options: Dict. the options for tracking (see greto.tracking.track_event)
    """
    try:
        result, stats = track_event(event, return_stats=True, **options)
        return result, stats
    except Exception as e:
        print(f"Error processing event: {e}")
        return None, None


def worker(input_queue: Queue, output_queue: Queue, stats_queue: Queue, **options):
    """
    Worker watches input queue for tasks and then processes them

    Args
    ----
    input_queue: Queue. queue containing g-ray events from the loader
    output_queue: Queue. queue containing mode1 or mode1x struct bytes to write
        to the output file
    stats_queue: Queue. queue for collecting statistics
    options: Dict. the options for tracking (see greto.tracking.track_event)
    """
    accumulated_stats = defaultdict(int)
    while True:
        event = input_queue.get()
        if event is None:  # Sentinel value to indicate end of queue
            break
        result, stats = process_event(event, **options)
        if result is not None:
            output_queue.put(result)
            if stats:
                for key, value in stats.items():
                    accumulated_stats[key] += value
    output_queue.put(None)  # Signal that this worker is done
    stats_queue.put(dict(accumulated_stats))  # Send accumulated stats


def stats_collector(stats_queue: Queue, num_workers: int):  # , log_filename: str):
    """
    Collects statistics from workers and writes them to a log file

    Args
    ----
    stats_queue: Queue. queue for collecting statistics
    num_workers: int. number of workers
    log_filename: str. path to log file
    """
    total_stats = defaultdict(int)
    workers_done = 0
    while workers_done < num_workers:
        stats = stats_queue.get()
        if stats is not None:
            for key, value in stats.items():
                total_stats[key] += value
            workers_done += 1
    pad = []
    pad_counts = []
    crystal = []
    crystal_counts = []
    for key, val in total_stats.items():
        if key.startswith("pad"):
            pad.append(int(str(key).split()[-1]))
            pad_counts.append(val)
        if key.startswith("crystal"):
            crystal.append(int(str(key).split()[-1]))
            crystal_counts.append(val)
    print("Pads")
    for p, pc in zip(pad, pad_counts):
        print(f"pad {p:3d} accounted for {pc:10d} hits")
    print()
    print("crystals hits")
    print()
    for c, cc in zip(crystal, crystal_counts):
        print(f"crystal {c:3d} had {cc:10d} hits")
    print()
    print(f"Clusters {total_stats['clusters']}")
    # with open(log_filename, 'w') as log_file:
    #     for key, value in total_stats.items():
    #         log_file.write(f"{key}: {value}\n")


def writer(output_queue: Queue, output_filename: str, num_workers: int):
    """
    Writer watches output queue for data and then writes it

    Args
    ----
    output_queue: Queue. queue containing mode1 or mode1x struct bytes to write
        to the output file
    output_filename: str. path to output file
    num_workers: int. number of workers
    """
    with open(output_filename, mode="wb") as output_file:  # Open file
        workers_done = 0
        while workers_done < num_workers:  # While there are still workers
            result = output_queue.get()  # Grab output from queue
            if result is None:  # Worker sent back a None...
                workers_done += 1  # ... indicating complete
            else:
                output_file.write(result)  # Write output to file


def track_async(
    input_filename: str,
    output_filename: str,
    options_filename: str | dict,
    queue_size: int = 100,
):
    """
    Track input_filename and save tracked output to output_filename using
    options_filename options

    Args
    ----
    input_filename: str. path to input file (mode2 or mode1x)
    output_filename: str. path to output file (mode1 or mode1x)
    options_filename: str. path to options .yaml file or dictionary of options
    queue_size: int. number of events to keep on the input queue
    """
    start_time = time.time()
    if isinstance(options_filename, dict):
        options = options_filename
    else:
        options = load_options(options_filename)

    print(f"Detector is set to {options.get('DETECTOR', 'defaulting to GRETINA')}")
    default_config.set_detector(options.get("DETECTOR", "GRETINA"))

    if options.get("SAVE_INTERMEDIATE", False):
        print("Tracking will save an intermediate tracked format instead of mode1")
    elif options.get("SAVE_EXTENDED_MODE1", False):
        print("Tracking will save an extended mode1 data format with all interactions")

    options = load_models(options)

    num_workers = options.get("NUM_PROCESSES")
    if num_workers is None:
        num_workers = cpu_count()

    input_queue = Queue(maxsize=queue_size)
    output_queue = Queue()
    stats_queue = Queue()

    # Start worker processes
    workers: List[Process] = []
    for _ in range(num_workers):
        p = Process(
            target=worker, args=(input_queue, output_queue, stats_queue), kwargs=options
        )
        p.start()
        workers.append(p)

    # Start writer process
    writer_process = Process(
        target=writer, args=(output_queue, output_filename, num_workers)
    )
    writer_process.start()

    # Start stats collector process
    stats_process = Process(
        target=stats_collector, args=(stats_queue, num_workers)  # , log_filename)
    )
    stats_process.start()

    # Read and queue events
    with open(input_filename, mode="rb") as input_file:
        if options["READ_MODE1"]:
            loader = mode1_loader(
                input_file,
                detector=default_config,
                print_formatted=options["PRINT_FORMATTED"],
            )
        else:
            loader = mode2_loader(
                input_file,
                time_gap=options["COINCIDENCE_TIME_GAP"],
                detector=default_config,
                global_coords=options["GLOBAL_COORDS"],
                print_formatted=options["PRINT_FORMATTED"],
            )

        filesize = get_file_size(input_file)
        processed_count = 0
        last_processed_count = 0
        last_update_time = time.time()

        for event in loader:
            if event is not None:
                processed_count += 1
                # Clean up the incoming event
                event = remove_zero_energy_interactions(event)
                event = pack_interactions(event, packing_distance=1e-8)
                input_queue.put(event)
                if time.time() - last_update_time > 60:  # Update every minute
                    print(
                        f"Processed {processed_count} events. Elapsed time: "
                        + f"{time.time() - start_time:.2f} seconds; "
                        + f"rate = {(processed_count - last_processed_count) / (time.time() - last_update_time):5.2f} events/second"
                        + f"\n{input_file.tell():11d}/{filesize:11d} "
                        + f"{100*input_file.tell()/filesize:10.10f}%"
                    )
                    last_update_time = time.time()
                    last_processed_count = processed_count
                if options["PARTIAL_TRACK"]:
                    if processed_count > options.get("PARTIAL_TRACK", float("inf")):
                        print(
                            f"Hit event limit {options.get('PARTIAL_TRACK', float('inf'))}"
                            + " specified by 'PARTIAL_TRACK' option"
                        )
                        break

    # Signal workers to finish
    for _ in range(num_workers):
        input_queue.put(None)

    # Wait for all workers to finish
    for w in workers:
        w.join()

    # Wait for writer to finish
    writer_process.join()

    # Wait for stats collector to finish
    stats_process.join()

    print(f"Total time: {time.time() - start_time:.5f} seconds")
    print(
        f"Average speed: {processed_count/(time.time() - start_time):.5f} events/second"
    )
    print(f"Processed {processed_count} events")
    print(f"Output saved to {output_filename}")
