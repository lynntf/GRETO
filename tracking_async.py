import asyncio
import logging
import time

import aiofiles

from greto.event_class import Event
from greto.file_io import load_options, mode1_loader, mode2_loader
from greto.models import load_order_FOM_model
from greto.tracking import track_and_get_energy

logging.basicConfig(level=logging.INFO)


async def track(event: Event, **kwargs):
    """Track an event asynchronously"""
    return await asyncio.to_thread(track_and_get_energy, event, **kwargs)


async def process_event(
    event: Event,
    file_lock: asyncio.Lock,
    file_handle,
    **track_kwargs,
):
    """Get the tracking result and append it to the output file"""
    start_time = time.time()
    try:
        result = await track(event, **track_kwargs)
        async with file_lock:
            await file_handle.write(result)
        logging.info(
            "Event %s processed in %.5f seconds", event.id, time.time() - start_time
        )
    except Exception as e:
        logging.error("Error processing event: %s", e)


async def event_producer(input_filename, options, queue):
    """Produce events from the input file"""
    with open(input_filename, mode="rb") as input_file:
        if options["READ_MODE1"]:
            loader = mode1_loader(input_file)
        else:
            loader = mode2_loader(input_file)
        for event, _ in zip(loader, range(100)):
            await queue.put(event)
            logging.debug(f"Event added to queue")
    await queue.put(None)  # Signal end of events
    logging.info("All events added to queue")


async def event_consumer(queue, file_lock, file_handle, **options):
    """Consume events from the queue and process them"""
    while True:
        event = await queue.get()
        if event is None:
            queue.task_done()
            break
        await process_event(event, file_lock, file_handle, **options)
        queue.task_done()
        logging.debug(f"Queue size: {queue.qsize()}")


async def main(
    input_filename: str,
    output_filename: str,
    options_filename: str,
    max_concurrent_tasks: int = 20,
):
    """Do the main asynchronous tracking"""
    start_time = time.time()
    options = load_options(options_filename)
    file_lock = asyncio.Lock()

    if options["order_FOM_kwargs"].get("fom_method") == "model":
        if options["order_FOM_kwargs"].get("model_filename") is not None:
            order_model = load_order_FOM_model(
                options["order_FOM_kwargs"].get("model_filename")
            )
        else:
            raise ValueError("Provide model filename for ordering")
        options["order_FOM_kwargs"]["model"] = order_model

    queue = asyncio.Queue(maxsize=max_concurrent_tasks * 2)

    async with aiofiles.open(output_filename, mode="wb") as output_file:
        producer = asyncio.create_task(event_producer(input_filename, options, queue))
        consumers = [
            asyncio.create_task(
                event_consumer(queue, file_lock, output_file, **options)
            )
            for _ in range(max_concurrent_tasks)
        ]

        await producer
        await queue.join()

        for _ in range(max_concurrent_tasks):
            await queue.put(None)
        await asyncio.gather(*consumers)
    print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
    input_filename = "data12.mode2"
    output_filename = "test.mode1"
    options_filename = "data12_ML_model_ordering.yaml"
    asyncio.run(main(input_filename, output_filename, options_filename))
