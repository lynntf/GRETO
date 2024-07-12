"""Track using multiprocessing"""

import time

from joblib import Parallel, delayed

from greto.cluster_tools import pack_interactions, remove_zero_energy_interactions
from greto.detector_config_class import default_config
from greto.event_class import Event
from greto.file_io import load_options, mode1_loader, mode2_loader
from greto.models import load_order_FOM_model, load_suppression_FOM_model
from greto.tracking import track_event
from greto.utils import get_file_size


def process_event(event: Event, **kwargs):
    """Track each event"""
    try:
        start_time = time.time()
        result = track_event(event, **kwargs)
        print(f"Event {event.id} processed in {time.time() - start_time:.5f} seconds")
        return result
    except Exception as e:
        print(f"Error processing event: {e}")
        return None


def process_batch(batch, **options):
    """Track a batch of events"""
    results = []
    for event in batch:
        try:
            result = process_event(event, **options)
            results.append(result)
        except Exception as e:
            print(f"Error in process_batch: {e}")
    return results


def load_models(options: dict) -> dict:
    """Load the machine learning models for tracking"""
    order_model = None
    if options["order_FOM_kwargs"].get("fom_method") == "model":
        if options["order_FOM_kwargs"].get("model_filename") is not None:
            order_model = load_order_FOM_model(
                options["order_FOM_kwargs"].get("model_filename")
            )
        else:
            raise ValueError("Provide model filename for ordering")
    options["order_FOM_kwargs"]["model"] = order_model

    secondary_order_model = None
    if options["secondary_order_FOM_kwargs"].get("fom_method") == "model":
        if options["secondary_order_FOM_kwargs"].get("model_filename") is not None:
            secondary_order_model = load_order_FOM_model(
                options["secondary_order_FOM_kwargs"].get("model_filename")
            )
        else:
            raise ValueError("Provide model filename for secondary ordering")
    options["secondary_order_FOM_kwargs"]["model"] = secondary_order_model

    eval_model = None
    if options["eval_FOM_kwargs"].get("fom_method") == "model":
        if options["eval_FOM_kwargs"].get("model_filename") is not None:
            eval_model = load_suppression_FOM_model(
                options["eval_FOM_kwargs"].get("model_filename")
            )
        else:
            raise ValueError("Provide model filename for evaluation")
    options["eval_FOM_kwargs"]["model"] = eval_model
    return options


def main(
    input_filename: str,
    output_filename: str,
    options_filename: str,
    batch_size: int = 100,
):
    """Do the main parallel tracking"""
    start_time = time.time()
    options = load_options(options_filename)

    print(f"Detector is set to {options.get('DETECTOR', 'defaulting to GRETINA')}")
    default_config.set_detector(options.get("DETECTOR", "GRETINA"))

    if options.get("SAVE_INTERMEDIATE", False):
        print("Tracking will save an intermediate tracked format instead of mode1")
    elif options.get("SAVE_EXTENDED_MODE1", False):
        print("Tracking will save an extended mode1 data format with all interactions")

    options = load_models(options)

    max_concurrent_tasks = options["NUM_PROCESSES"]

    with open(input_filename, mode="rb") as input_file, open(
        output_filename, mode="wb"
    ) as output_file:
        filesize = get_file_size(input_file)  # Filesize in bytes

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

        batch = []
        processed_count = 0
        last_update_time = time.time()

        for event, _ in zip(loader, range(1000)):  # TODO - remove 100 event limit
            if event is not None:
                # Clean up the incoming event
                event = remove_zero_energy_interactions(event)
                event = pack_interactions(event, packing_distance=1e-8)
                batch.append(event)
            if len(batch) == batch_size:
                processed_count += len(batch)
                try:
                    results = Parallel(n_jobs=max_concurrent_tasks)(
                        delayed(process_event)(ev, **options) for ev in batch
                    )
                    for result in results:
                        if result is not None:
                            output_file.write(result)
                except Exception as e:
                    print(f"Error in parallel processing: {e}")
                if time.time() - last_update_time > 60:  # Update every minute
                    print(
                        f"Processed {processed_count} events. Elapsed time: "
                        + f"{time.time() - start_time:.2f} seconds"
                        + f"\n{input_file.tell():10d}/{filesize:10d} {100*input_file.tell()/filesize:10.10f}%"
                    )
                    last_update_time = time.time()
                batch = []
            if options["PARTIAL_TRACK"]:
                if processed_count > options.get("PARTIAL_TRACK", float("inf")):
                    print(
                        f"Hit event limit {options.get('PARTIAL_TRACK', float('inf'))}"
                        + " specified by 'PARTIAL_TRACK' option"
                    )
                    break

        # Process any remaining events
        if batch:
            processed_count += len(batch)
            try:
                results = Parallel(n_jobs=max_concurrent_tasks)(
                    delayed(process_event)(ev, **options) for ev in batch
                )
                for result in results:
                    if result is not None:
                        output_file.write(result)
            except Exception as e:
                print(f"Error in processing remaining events: {e}")

    print(f"Total time: {time.time() - start_time:.5f} seconds")
    print(f"Processed {processed_count} events")


if __name__ == "__main__":
    input_filename = "data12.mode2"
    output_filename = "test.mode1"
    options_filename = "data12_ML_model_ordering.yaml"
    main(input_filename, output_filename, options_filename)
