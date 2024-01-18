"""
Script for streamlined tracking.
"""
import argparse

from gamma_ray_tracking.tracking import load_and_track_files


def track_with_args():
    """
    Read in details from the command line and perform tracking
    """
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("mode2_filename", help="Path to the input mode2 file.")
    parser.add_argument("mode1_filename", help="Path to the output mode1 file.")
    parser.add_argument(
        "options_file", nargs="?", help="Path to the JSON/YAML options file."
    )
    args = parser.parse_args()

    load_and_track_files(args.mode2_filename, args.mode1_filename, args.options_file)


if __name__ == "__main__":
    track_with_args()
