"""
Script for streamlined tracking.
"""

import argparse
import yaml

from greto.file_io import mode1x_new_fom, convert_mode1_extended


def convert_with_args():
    """
    Read in details from the command line and performs a conversion from a
    mode1x file to a mode1(x) file, possibly with a new FOM.
    
    Reads in arguments from the command line:
        - mode1x_in_filename
        - mode1x_out_filename
        - options_filename
    """
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("mode1x_in_filename", help="Path to the input mode1x file.")
    parser.add_argument("mode1x_out_filename", help="Path to the output mode1(x) file.")
    parser.add_argument(
        "options_filename", nargs="?", help="Path to the JSON/YAML options file."
    )
    args = parser.parse_args()

    if args.options_filename is not None:
        with open(args.options_filename, "r", encoding="utf-8") as options_file:
            loaded_options = yaml.safe_load(options_file)
    else:
        loaded_options = {}

    reevaluate_FOM = loaded_options.get("reevaluate_FOM", False)  # Default to just conversion
    save_extended = loaded_options.get("save_extended", True)
    eval_FOM_kwargs = loaded_options.get("eval_FOM_kwargs", {})

    with open(args.mode1x_in_filename, "rb") as input_file:
        with open(args.mode1x_out_filename, "rb") as output_file:
            if reevaluate_FOM:
                mode1x_new_fom(
                    input_file, output_file, save_extended=save_extended, **eval_FOM_kwargs
                )
            else:
                convert_mode1_extended(input_file, output_file)


if __name__ == "__main__":
    convert_with_args()
