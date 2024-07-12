"""
Track using single thread

This script outlines the basic logic of the tracking process
"""

import argparse
import time

from greto.detector_config_class import default_config
from greto.event_class import Event
from greto.file_io import load_options, mode1_loader, mode2_loader
from greto.models import load_models, load_order_FOM_model
from greto.tracking import track_event


def track(event: Event, **kwargs):
    return track_event(event, **kwargs)


def process_event(
    event: Event,
    filename,
    **track_kwargs,
):
    """Get the tracking result and append it to the output file"""
    start_time = time.time()
    try:
        result = track(event, **track_kwargs)
        with open(filename, "ab") as file_handle:
            file_handle.write(result)
        print(f"Event {event.id} processed in {time.time() - start_time:.5f} seconds")
    except Exception as e:
        print(f"Error processing event: {e}")


def main(
    input_filename: str,
    output_filename: str,
    options_filename: str,
):
    """Do the main asynchronous tracking"""
    start_time = time.time()
    options = load_options(options_filename)

    print(f"Detector is set to {options.get('DETECTOR', 'defaulting to GRETINA')}")
    default_config.set_detector(options.get("DETECTOR", "GRETINA"))

    if options.get("SAVE_INTERMEDIATE", False):
        print("Tracking will save an intermediate tracked format instead of mode1")
    elif options.get("SAVE_EXTENDED_MODE1", False):
        print("Tracking will save an extended mode1 data format with all interactions")

    options = load_models(options)

    # with open(output_filename, mode="wb") as output_file:
    with open(input_filename, mode="rb") as input_file:
        if options["READ_MODE1"]:
            loader = mode1_loader(input_file)
        else:
            loader = mode2_loader(input_file)
        for count, event in enumerate(loader):
            if event is not None:
                process_event(event, output_filename, **options)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    # Read in details from the command line and perform tracking
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "input_filename", help="Path to the input (mode2, mode1x) file."
    )
    parser.add_argument(
        "output_filename", help="Path to the output (mode1, mode1x) file."
    )
    parser.add_argument(
        "options_file", nargs="?", help="Path to the JSON/YAML options file."
    )
    args = parser.parse_args()

    main(args.input_filename, args.output_filename, args.options_file)
