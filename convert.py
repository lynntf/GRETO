"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Script for conversion to mode1 files or re-evaluating the FOM of mode1x files.
"""

import argparse
import yaml

from greto.file_io import mode1x_new_fom, convert_mode1_extended
from greto.models import load_suppression_FOM_model


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

    with open(
        "greto/convert_default.yaml", "r", encoding="utf-8"
    ) as default_options_file:
        default_options = yaml.safe_load(default_options_file)

    loaded_options = default_options | loaded_options

    DETECTOR = loaded_options.get("DETECTOR", "gretina")  # Default to gretina
    SAVE_EXTENDED_MODE1 = loaded_options.get(
        "SAVE_EXTENDED_MODE1", False
    )  # Default to just conversion
    REEVALUATE_FOM = loaded_options.get(
        "REEVALUATE_FOM", False
    )  # Default to just conversion
    MONSTER_SIZE = loaded_options.get(
        "MONSTER_SIZE", 8
    )  # Value to determine if the cluster was not tracked
    eval_FOM_kwargs = loaded_options.get("eval_FOM_kwargs", {})

    eval_model = None
    if eval_FOM_kwargs.get("fom_method") == "model":
        if eval_FOM_kwargs.get("model_filename") is not None:
            eval_model = load_suppression_FOM_model(
                eval_FOM_kwargs.get("model_filename")
            )
        else:
            raise ValueError("Provide model filename for evaluation")
    eval_FOM_kwargs["model"] = eval_model

    with open(args.mode1x_in_filename, "rb") as input_file:
        with open(args.mode1x_out_filename, "wb") as output_file:
            if REEVALUATE_FOM:
                print(f"Applying new FOM evaluation to data from detector:{DETECTOR.capitalize()} using: {eval_FOM_kwargs}")
                mode1x_new_fom(
                    input_file,
                    output_file,
                    save_extended=SAVE_EXTENDED_MODE1,
                    detector_name=DETECTOR,
                    monster_size=MONSTER_SIZE,
                    debug = False,
                    **eval_FOM_kwargs,
                )
            elif not SAVE_EXTENDED_MODE1:
                print("Converting mode1x to mode1.")
                convert_mode1_extended(input_file, output_file, debug=False)
                print("Conversion complete.")
            else:
                print(
                    "Saving extended mode1 without evaluating a new FOM will produce an identical mode1x. Conversion no performed."
                )


if __name__ == "__main__":
    convert_with_args()
