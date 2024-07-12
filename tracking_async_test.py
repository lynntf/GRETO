"""Track using multiprocessing"""

import time
from collections import defaultdict
from multiprocessing import Process, Queue, cpu_count
from typing import List

from greto.cluster_tools import pack_interactions, remove_zero_energy_interactions
from greto.detector_config_class import default_config
from greto.event_class import Event
from greto.file_io import load_options, mode1_loader, mode2_loader
from greto.models import load_order_FOM_model, load_suppression_FOM_model
from greto.tracking import track_event
from greto.utils import get_file_size


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


def load_models(options: dict) -> dict:
    """
    Load the machine learning models for tracking

    Places models into the options dictionary

    Args
    ----
    options: dict. dictionary of options from the tracking .yaml file
    """

    # Model for ordering
    order_model = None
    if options["order_FOM_kwargs"].get("fom_method") == "model":
        if options["order_FOM_kwargs"].get("model_filename") is not None:
            order_model = load_order_FOM_model(
                options["order_FOM_kwargs"].get("model_filename")
            )
        else:
            raise ValueError("Provide model filename for ordering")
    options["order_FOM_kwargs"]["model"] = order_model

    # Model for secondary ordering
    secondary_order_model = None
    if options["secondary_order_FOM_kwargs"].get("fom_method") == "model":
        if options["secondary_order_FOM_kwargs"].get("model_filename") is not None:
            secondary_order_model = load_order_FOM_model(
                options["secondary_order_FOM_kwargs"].get("model_filename")
            )
        else:
            raise ValueError("Provide model filename for secondary ordering")
    options["secondary_order_FOM_kwargs"]["model"] = secondary_order_model

    # Model for evaluating the ordered g-rays
    eval_model = None
    if options["eval_FOM_kwargs"].get("fom_method") == "model":
        if options["eval_FOM_kwargs"].get("model_filename") is not None:
            eval_model = load_suppression_FOM_model(
                options["eval_FOM_kwargs"].get("model_filename")
            )
        else:
            raise ValueError("Provide model filename for evaluation")
    options["eval_FOM_kwargs"]["model"] = eval_model

    # Return updated options
    return options


def main(
    input_filename: str,
    output_filename: str,
    options_filename: str,
    queue_size: int = 100,
):
    """
    Track input_filename and save tracked output to output_filename using
    options_filename options

    Args
    ----
    input_filename: str. path to input file (mode2 or mode1x)
    output_filename: str. path to output file (mode1 or mode1x)
    options_filename: str. path to options .yaml file
    queue_size: int. number of events to keep on the input queue
    """
    start_time = time.time()
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
                if time.time() - last_update_time > 10:  # Update every minute
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


if __name__ == "__main__":
    input_filename = "data12.mode2"
    output_filename = "test.mode1"
    options_filename = "test.yaml"
    main(
        input_filename,
        output_filename,
        options_filename,
    )
