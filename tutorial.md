# Tracking tutorial

## Installation
1. Git clone the GRETO repository
    ```bash
    git clone https://github.com/lynntf/GRETO.git
    ```
2. `cd` into the code directory
    ```bash
    cd GRETO
    ```
3. Create Python virtual environment and install dependencies using `requirements_tracking.txt` (required for tracking) or `requirements.txt` (for development and plotting)
    * Using conda:
        ```bash
        conda create --name greto_env --file requirements_tracking.txt
        ```
        or
        ```bash
        conda create --name greto_env --file requirements.txt
        ```
4. Activate virtual environment to access dependencies
    ```bash
    conda activate greto_env
    ```

## Using the software
### Tracking
Tracking is done using the script `track.py` that takes in some command line arguments for input, output, and options:
```bash
python -u track.py input.mode2 output.mode1 example_options.yaml > track.log
```
The option `-u` specifies that the output from python is un-buffered (writes to `track.log` immediately).

An example `options.yaml`:
```yaml
---
# Example settings for gamma-ray tracking

DETECTOR: gretina               # Available detectors are "GRETINA", "GRETA" (identical to "GRETINA"), and "AGATA"
COINCIDENCE_TIME_GAP: &tg 40    # Units of 10 ns; Used for grouping recorded crystal events in time
NUM_PROCESSES: 3                # Number of processors to use for multi-processing (should be changed depending on the machine)
VERBOSITY: 2                    # Level of console output;
TIMEOUT_SECONDS: 15             # Timeout limit in seconds for multiprocessing processes
MAX_HIT_POINTS: 300             # Maximum number of hit points for constructed events (experiment/data dependent)
MONSTER_SIZE: &monster 8        # Maximum size of a cluster; clusters larger than the "monster" size are not tracked
SAVE_EXTENDED_MODE1: false      # Save a mode1 file with all interactions recorded
REEVALUATE_FOM: false           # Reevaluate the FOM values according to the eval_FOM_kwargs; Used in conversion, not used for tracking
order_FOM_kwargs:               # Arguments for ordering interactions (recorded in mode1)
  # fom_method: selected          # FOM method for ordering; Available methods are "oft"/"agata", "aft"/"angle", "selected" (ML optimized for ordering), "model"
  # model_filename: null          # Filename of the model
  fom_method: model
  model_filename: models/ordering/N2000_lp_nonnegFalse_C10000_cols-oft_fast_tango_width5.json
  width: 5                      # How many interactions to consider in the forward direction in the enumeration of possible tracks
  stride: 2                     # How many interactions to accept after each enumeration step
  max_cluster_size: *monster    # Maximum cluster size for tracking (should be the same as the monster size, but can be different)
eval_FOM_kwargs:                # Suppression FOM. What FOM should be recorded in mode1
  # fom_method: angle             # FOM method for recording in mode1
  # model_filename: null          # Filename of the model
  # singles_method: depth         # Singles method for recording in mode1; Available methods are "depth" (identical to AFT treatment), "range" (similar to AFT treatment using a continuous function to determine ranges), "probability" (range probability is returned), "continuous" (linear attenuation * distance)
  fom_method: model
  model_filename: models/suppression/N10000_sns-logistic_pca-0.95_order-model.pkl
  max_cluster_size: *monster    # Maximum cluster size for tracking, determines if the gamma ray is considered tracked or not (should match the tracking options)
cluster_kwargs:                 # How should interactions be clustered?
  alpha_degrees: 20             # Cone clustering alpha in degrees
  time_gap: *tg                 # Time gap for clustering; interactions with timestamps differing by more than the time gap have the distance between them set to infinity (can still be clustered together if there is an interaction bridging the time gap)

```
Default options are used for any options that are not provided. Default options are in `GRETO/greto/track_default.yaml`. Some options that are particularly important in the `options.yaml` file are:
- `DETECTOR`: the detector that the data is coming from. This specifies the detector geometry and crystal coordinate conversions
- `SAVE_EXTENDED_MODE1`: save a mode1-like output (`mode1x`) that includes information about *all* interactions for each &gamma;-ray (not just the first two for each &gamma;-ray). The GEB header indicates type `33`.
    - Q: Why might you want to use this over just outputting mode1 data?
    - A: The extended mode1 file (`mode1x`) is completely tracked, but contains information that still allows further processing. For example, we can apply a different FOM for suppression than what was originally in the `eval_FOM_kwargs`, possibly improving suppression without completely re-tracking.
- `cluster_kwargs:alpha_degrees`: the angular distance used for clustering interaction points. This is dependent on the data.
- `*_FOM_kwargs:fom_method`: the method used to order interactions:
    - `aft`: Argonne Forward Tracking
    - `angle`: same as `aft`
    - `oft`: Orsay Forward Tracking
    - `agata`: same as `oft`
    - `selected`: an early, hardcoded ML model
    - `model`: an ML model. Provide the relative path to the model in `model_filename`.
        - Although the XGBoost models should perform better overall, I haven't been able to get them to be fast enough to use.
- `*_FOM_kwargs:singles_method`: the method used to evaluate single interactions. This is not used at all if an ML model is provided for suppression
    - `depth`: depth based rejection using interpolated values from `chat` file
    - `range`: depth based rejection using continuously computed values. Additional keyword arguments can be provided for this:
        - `singles_range`: the probability range at which rejection happens (default value is `0.95`; a distance at which we expect 95% of singles to have interacted already)
        - `singles_penalty_min`: minimum FOM assigned to the single (default value is `0.0`)
        - `singles_penalty_max`: "maximum" FOM assigned to the single (default value is `1.85`)
    - `continuous`: a continuous penalty based on distance and linear attenuation. Same additional keyword arguments as `range`
    - `probability`: cumulative probability of the ray traveling further than the distance it traveled. Same additional keyword arguments as `range`

### Converting from mode1x to mode1
`mode1x` is an extended `mode1` format that includes all of the interactions for each tracked event. The `mode1` file contains the first two interactions only (or the first and one empty interaction for single interactions).

Converting from `mode1x` to `mode1` is handled by the `convert.py` script that takes similar command line arguments as the `track.py` script. It can also be used to reevaluate the reported FOM and save as a `mode1` or `mode1x` again. This conversion step can be skipped in general by providing the desired output FOM (`eval_FOM_kwargs`) to the tracking script, but outputting a `mode1x` allows for the suppression FOM to be changed later without tracking again.
```bash
python -u convert.py input.mode1x output.mode1 example_options.yaml > convert.log
```
or without any reevaluation
```bash
python -u convert.py input.mode1x output.mode1 > convert.log
```

The `options.yaml` file can be the same as used for tracking, but the options file does not need all of the same fields as the tracking options because the convert script doesn't track. There is one extra option for converting that specifies of the FOM is going to be reevaluated or not (`REEVALUATE_FOM`). If the FOM isn't going to be reevaluated, then the default options will suffice and an options file doesn't need to be provided. Conversion from `mode1x` to `mode1` without reevaluating a FOM is just moving struct data around and is very fast.

### Available ML models
All models are stored in the `GRETO/models` directory of the git repository organized into `ordering` or `suppression`. Suppression models cannot currently be used for tracking. Ordering models cannot currently be used for suppression.

The hard coded FOM methods can be used for either ordering or suppression or both. A singles method should be provided to get FOM values for single interactions.

The XGBoost (boosted tree) models are, in general, much more accurate for ordering (but significantly slower). However, these models are probably NOT usable at all for suppression because the training data has peaks that these models will simply use to cheat training (the model will learn to identify the peaks in the training data using energy or something correlated to energy without learning how to actually suppress anything). Using a white-noise spectrum for training would allow these models to be used and would probably produce the best results.

The python based suppression models are actually two models that are combined: a model for single interactions and a model for non-single interactions (`sns`: singles and non-singles). Each model takes input data, rescales the data (based on training data to have mean zero and standard deviation of 1), does a PCA (principal component analysis) decomposition, applies a linear model (either a linear regression or a logistic regression), and then combines the two model outputs using a third linear model (same type: linear or logistic as before). Linear models produce FOM values of all sorts, logistic models produce values between zero and one, zero is high confidence of inclusion in the spectrum and one is low confidence of inclusion.

The code that splits the singles and non-singles is objectively bad right now (really dumb software solution) that should be changed in the future to just measure how many intersections are there, but this unfortunately requires a lot more work.

The orderings models are stored as plain text (except the XGBoost ones that are a text+binary format). Some feature weights are small and can be set to zero without any problems (less than `1e-8` is basically numerical zero for the process that came up with the weights). Each feature is scaled and then multiplied by the scale-free feature weight. Because these models are trained to just be used to compare different interaction orders, the number that they spit out is not necessarily useful for suppression.

The model file name contains the information about how it was trained. All of them were trained using the multiplicity-30 data. Ordering models ignore the fake events and are trained on individual gamma-rays. Suppression models are trained using the true ordering, expected ordering from a model output, and expected ordering after clustering the fake multiplicity-30 events.

The method used to define the model is either LP (linear program), MILP (mixed integer linear program; integrality is relaxed though so this is just a different linear program), SVM (support vector machine), LR (logistic regression), or an XGBoost XGBRanker. The LP and SVM results seem to be the best of the linear models. The models are trained using different subsets of features (this is roughly how the computational load of using each model is controlled) such as "aft_true" which returns weights for an optimized AFT method defined by two features. Models trained using "oft_fast_tango" seem to perform the best. This is a combination of OFT, "fast": aft and some other easy to compute features, and tango variants of features (tango energy used as incoming energy). Tango features use tango computed incoming energy if it makes sense physically, otherwise they are identical to other features using an energy sum. The tango energy here is only used for ordering and is not reported in any output.

## Warnings
### Escape probability feature
The probability of a gamma ray escaping is calculated assuming the detector is a sphere (not a shell, just a sphere with some outer radius) and using the tango energy (if the tango energy doesn't make sense, probability is set to zero). This feature sometimes produces warnings about integration accuracy. I believe that all of the relevant problems with this computation have been smoothed out, so those warnings can be safely ignored (a warning indicates a potential problem with the data decomposition).

### Other warnings
Some other warnings seem to be typical. Warnings about divide by zero when computing a logarithm seem to be a result of bad signal decomposition so eliminating that data should probably remove the warnings. These warnings get doubled when the tango energy is used for some features (warning for energy sum and warning for tango energy estimate, which may be identical to the energy sum).

## Example usage
Order, suppress, and output `mode1` (`SAVE_EXTENDED_MODE1` is set to `false`)
```bash
python -u track.py input.mode2 output.mode1 example_options.yaml > track.log
```

Order, suppress, and output `mode1x` (`SAVE_EXTENDED_MODE1` is set to `true`)
```bash
python -u track.py input.mode2 output.mode1x different_options.yaml > track.log
```
`output.mode1x` can now be re-evaluated (`REEVALUATE_FOM` is set to `true`)
```bash
python -u convert.py output.mode1x output.mode1 new_example_options.yaml > convert.log
```

## Re-building the machine learning models
Ordering models
```bash
python -u build_ordering_models.py > build_models.log
```
There are a variety of options within the `build_ordering_models.py` script that dictate how the models are put together.

Suppression models (requires one of the ordering models to run)
```bash
python -u build_classification_models.py > build_models.log
```

The suppression models require ordering &gamma;-rays using an ordering model. This is because we want the suppression model to be trained on the expected output of ordering