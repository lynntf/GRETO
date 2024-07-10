import time

from greto.event_class import Event
from greto.file_io import load_options, mode1_loader, mode2_loader
from greto.models import load_order_FOM_model
from greto.tracking import track_and_get_energy


def track(event: Event, **kwargs):
    return track_and_get_energy(event, **kwargs)


def process_event(
    event: Event,
    filename,
    **track_kwargs,
):
    """Get the tracking result and append it to the output file"""
    start_time = time.time()
    try:
        result = track(event, **track_kwargs)
        with open(filename, 'ab') as file_handle:
            file_handle.write(result)
        print(f"Event {event.id} processed in {time.time() - start_time:.5f} seconds")
    except Exception as e:
        print(f"Error processing event: {e}")


def main(
    input_filename: str,
    output_filename: str,
    options_filename: str,
    max_concurrent_tasks: int = 10,
):
    """Do the main asynchronous tracking"""
    start_time = time.time()
    options = load_options(options_filename)

    if options["order_FOM_kwargs"].get("fom_method") == "model":
        if options["order_FOM_kwargs"].get("model_filename") is not None:
            order_model = load_order_FOM_model(
                options["order_FOM_kwargs"].get("model_filename")
            )
        else:
            raise ValueError("Provide model filename for ordering")
        options["order_FOM_kwargs"]["model"] = order_model

    count = 0
    max_count = 100
    # with open(output_filename, mode="wb") as output_file:
    with open(input_filename, mode="rb") as input_file:
        if options["READ_MODE1"]:
            loader = mode1_loader(input_file)
        else:
            loader = mode2_loader(input_file)
        for event, _ in zip(loader, range(max_count)):
            count += 1
            process_event(event, output_filename, **options)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    input_filename = "data12.mode2"
    output_filename = "test.mode1"
    options_filename = "data12_ML_model_ordering.yaml"
    main(input_filename, output_filename, options_filename)
