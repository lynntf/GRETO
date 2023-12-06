# Gamma Ray Energy Tracking Optimization (GRETO)

The Gamma Ray Energy Tracking Array (GRETA) is a 4&pi; spherical shell, gamma-ray detector array used for gamma-ray spectroscopy. GRETA is able to determine the 3D position and energy of various gamma-ray interactions. The accuracy of the positions and energies are governed by the Pulse Shape Analysis (PSA) process, but the accuracy of the produced spectrum is governed by solving the gamma-ray tracking problem, for which we aim to improve the solution accuracy.

An overview of gamma-ray tracking in 4&pi; tracking arrays can be found in [**Tracking &gamma; rays in highly segmented HPGe detectors: A review
of AGATA and GRETINA**, A. Korichi and T. Lauritsen 2019](https://doi.org/10.1140/epja/i2019-12787-1).

## Tracking in brief

GRETINA/GRETA is designed to capture simultaneous emissions of multiple gamma-rays. Its open design (as opposed to the cellular design of GammaSphere) allows more gamma-rays to be kept for the final measured spectrum (GammaSphere relies on rejection of gamma-rays that exit the cell they enter). Although this allows more gamma-rays to be maintained for the final spectrum, detected gamma-rays must now be algorithmically separated from one another (where they were physically separated from one another by cell walls in GammaSphere). Many of the interactions detected by GRETINA/GRETA are Compton scattering interactions where an incoming gamma-ray elastically "bounces off" of an electron, depositing a portion of its energy and continuing on as a reduced energy gamma-ray. These scattering interactions obey the [Compton Scattering Formula (CSF)](https://en.wikipedia.org/wiki/Compton_scattering#Description_of_the_phenomenon). The goal of tracking is to find the causal sequence of measured interactions that best matches the actual gamma-ray emission, typically by evaluating the agreement of measured data with the CSF and known [gamma-ray cross-sections](https://en.wikipedia.org/wiki/Gamma_ray_cross_section).

One of the main challenges in tracking is that it is unknown whether or not the interactions that are measured are the result of a gamma-ray that deposits all of its energy in the active part of the array, or only deposits only some of its energy (exits the array or deposits energy in a non-active part of the detector). If we can successfully remove incomplete energy deposited gamma-rays from the data, we can remove background noise from the final spectrum. This removal process is called *Compton suppression*.

The energy of an emitted gamma-ray is computed by summing together the energies of the constituent interactions. Another key challenge is determining which interactions belong to which gamma-ray. Mixing interactions from different gamma-rays will essentially shift energy values into the background noise of the spectrum.

## Description

This software is designed to load gamma-ray events into a python readable format (from [typical GEB data types type 1 (mode2) and type 3 (mode1)](https://gretina.lbl.gov/tools-etc/gebheaders) and some other GEANT4 generated outputs). Given time sorted mode2 data (GRETINA/GRETA output from PSA), the software will build coincidences of gamma-ray interactions stored in `Interaction` objects [together, these comprise a single `Event` object or a `Coincidence` object (a stripped down `Event`); `Interaction` objects are stored in `Event.points` (including the detector origin/target) or in `Event.hit_points` (excluding the detector origin/target)]. Interactions from a single event must be tracked before they can be interpreted as part of a gamma-ray.

When reading interactions and events, a detector configuration must be provided. Default configurations for AGATA and GRETA are provided as `DetectorConfig(detector="agata")` and `DetectorConfig(detector="greta")` respectively. This can also be specified in a tracking `.yaml` or `.json` configuration file.

Given an `Event`, tracking is performed by first clustering interactions into separate gamma-rays.

> - `Event.cluster_linkage`, an alias for `cluster_tools.cluster_linkage`, will create clusters based on a hierarchical clustering method; the default method is a cone-clustering method where `Interaction` objects are grouped by their angle with respect to the detector center/target. Other methods are available for [linkages](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) and [pairwise distance calculations](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist) of which the default is a `cosine` distance.
> - One additional linkage not present in Scipy is an asymmetric linkage `asym_hier_linkage` or `directional`. This linkage type only allows clusters to grow from the head or tail of a cluster and creates the cluster as a chain of interactions (this is often undesired in generic hierarchical clustering, but matches the behavior of Compton scattered gamma-rays). This type of linkage allows for a small improvement in clustering accuracy with no change in clustering computational cost.
> - One additional distance metric is the `germanium` distance which indicates the pairwise length of detector material (which is germanium) between interactions. This assumes a geometrically simpler detector that is an exact spherical shell of material.

Clustered interactions are then put into a causal order with respect to a Figure-Of-Merit (FOM) that acts as an objective function. The order with the best (minimal or maximal) FOM determines the best causal order of interactions. Different FOMs have different performance with respect to ordering the interactions.

The methods for optimal FOM design are able to optimize this FOM for ordering. Given various different FOMs/FOM elements/features, an optimal FOM that is a function of them can be determined by framing ordering as a Learning-To-Rank (LETOR) problem. Various LETOR methods exist for determining ranking functions (essentially a FOM) that will compare different interaction orders. Given a method for comparing orders (e.g., a greedy comparison where the compared orders are only for the first few interactions, or a complete enumeration of all `N`-factorial orders), the code can construct features (FOMs and FOM elements) and then determine a model that combines these features such that the model selects the correct order as often as possible.

> *Notes*:
>
> - Not all features are created equal in terms of computational complexity and ordering ability.
> - An ensemble of ranking models is more likely to have robust performance. More specific models can be incorporated into an ensemble (e.g., energy specific models)
> - Learning a ranking model is dependent on the data used to train it, in this case a specific data set of simulated gamma-ray events from GEANT4. Other training data may produce more reliable ordering.

Because this is a general approach (i.e., given alternatives to compare to, design a model that compares them optimally), different approaches to clustering may also be possible. This software can produce features for individual clusters or groups of clusters. In addition, given other solution methods (e.g., [**Gamma-ray track reconstruction using graph neural networks**, Andersson and Bäck 2023](https://doi.org/10.1016/j.nima.2022.168000)), additional FOM elements can be constructed and incorporated. That is to say, individual methods for solving the tracking problem can be combined (somewhat) arbitrarily to increase overall performance and new tracking methods can be considered additive (instead of alternative) in performance.

Currently, a single LETOR derived ordering method (designated as the `selected` FOM) is included in this code although others are planned (optimizing existing FOMs by weighing FOM elements; optimizing for computational complexity of the features; optimizing for greedy approaches).

Clustering provides a python `dict` with keys indicating the cluster ID (usually integers) and values containing the ordered interaction IDs. Tracking a an event that has been clustered will produce a new clustering (dictionary) with reordered values.

## Installation

Currently, the only installation method is to clone the repository into the working directory.

This package requires Python 3.10 or newer. Consider creating a virtual environment with the name `gamma`:

```bash
conda create -n gamma python>3.10 scipy numpy
```

Activate the new `gamma` virtual environment

```bash
conda activate gamma
```

Some of the plotting functions in the `cluster_viz` module have additional dependencies, but these are not necessary for tracking:

```bash
conda install -c conda-forge matplotlib seaborn plotly cartopy pandas
```

Optimization methods for tracking require additional dependencies, again, these are not necessary for tracking:

```bash
conda install -c conda-forge cvxpy sklearn pyomo
```

Consider installing Jupyter and tqdm as well:

```bash
conda install jupyter tqdm
```

Deactivate the virtual environment:

```bash
conda deactivate
```

### Complete installation

```bash
conda create -n gamma -c conda-forge python>3.10 scipy numpy matplotlib seaborn plotly cartopy pandas cvxpy sklearn pyomo jupyter tqdm
```

## Usage

All of the following examples require that the `gamma_ray_tracking` folder is in the working directory.

### Tracking experimental data example

Mode2 data can be tracked directly using the command line:

```bash
python track.py path/to/mode2/data/file path/to/mode1/output/file path/to/tracking/options/file
```

The tracking options file is a `.yaml` or `.json` file containing options for tracking. An example tracking `.yaml` options file for Cobalt-60 source data is:

```yaml
---
# Settings for gamma-ray tracking
# Example for Cobalt-60

DETECTOR: gretina               # Available detectors are "GRETINA", "GRETA" (identical to "GRETINA"), and "AGATA"
COINCIDENCE_TIME_GAP: &tg 40    # Units of 10 ns; Used for grouping recorded crystal events in time
NUM_PROCESSES: 3                # Number of processors to use for multi-processing (should be changed depending on the machine)
SECONDARY_ORDER: false          # Track twice; Used for ensuring comparison between the same data
VERBOSITY: 2                    # Level of console output
TIMEOUT_SECONDS: 15             # Timeout limit in seconds for multiprocessing processes
MAX_HIT_POINTS: 100             # Maximum number of hit points for constructed events (experiment/data dependent)
GLOBAL_COORDS: false            # If the data is provided with global coordinates already, skip the transformation from crystal coordinates to global
MONSTER_SIZE: &monster 8        # Maximum size of a cluster; clusters larger than the "monster" size are not tracked
PARTIAL_TRACK: false            # Should tracking bail after determining the first interaction points? (Possibly useful when performing double tracking because only the first interactions are recorded in mode1 data)
SAVE_INTERMEDIATE: false        # Save an intermediate file format instead of mode1 (events and clusters stored by pickle)
SAVE_EXTENDED_MODE1: false      # Save a mode1 file with all interactions recorded
order_FOM_kwargs:               # Arguments for ordering interactions (recorded in mode1)
  fom_method: aft               # FOM method for ordering; Available methods are "oft"/"agata", "aft"/"angle", "selected" (ML optimized for ordering)
  singles_method: depth         # How to handle single interactions for ordering (not ordered)
  width: 5                      # How many interactions to consider in the forward direction in the enumeration of possible tracks
  stride: 2                     # How many interactions to accept after each enumeration step
  max_cluster_size: *monster    # Maximum cluster size for tracking (should be the same as the monster size, but can be different)
secondary_order_FOM_kwargs:     # If ordering twice (double tracking), reorder the data for recording a FOM as if ordered in this way in mode1
  fom_method: angle             # FOM method for ordering
  singles_method: depth         # How to handle single interactions for ordering (not ordered); this is not needed for ordering, but is used in the final mode1 output for singles
  width: 5                      # How many interactions to consider in the forward direction in the enumeration of possible tracks
  stride: 2                     # How many interactions to accept after each enumeration step
  max_cluster_size: *monster    # Maximum cluster size for tracking (should be the same as the monster size, but can be different)
eval_FOM_kwargs:                # What FOM should be recorded in mode1
  fom_method: angle             # FOM method for recording in mode1
  singles_method: depth         # Singles method for recording in mode1; Available methods are "depth" (identical to AFT treatment), "range" (similar to AFT treatment using a continuous function to determine ranges), "probability" (range probability is returned), "continuous" (linear attenuation * distance)
cluster_kwargs:                 # How should clustering of interaction points be accomplished?
  alpha_degrees: 20             # Cone clustering alpha in degrees
  time_gap: *tg                 # Time gap for clustering; interactions with timestamps differing by more than the time gap will (usually) not be clustered together by setting the effective distance between them to infinity
```

### Experimental data interactive example

For interactive uses of experimental data, we need to load the data into the `Event` objects used by the package:

```python
import gamma_ray_tracking as gr
mode2_filename = 'path/to/mode2/file.gtd'
detector_name = 'gretina'
events = []
with open(mode2_filename, 'rb') as mode2_file:
    for event in gr.file_io.mode2_loader(mode2_file, detector=detector_name):
        events.append(event)  # we probably do not want to actually retain all of these events, but rather process them as they are read in and discard the event objects after processing
```

These created events can be manipulated as below in the simulated data example (as either a list of events, which may be extremely large, as above, or individually).

Reading in raw mode1/mode2 data is done using the `GEBdata_file` class:

```python
import gamma_ray_tracking as gr
mode2_1_filename = 'path/to/mode2/file.gtd'
detector_name = 'gretina'
events = []
with open(mode2_1_filename, 'rb') as mode2_1_file:
    mode2_1 = gr.file_io.GEBdata_file(mode2_1_file, detector=detector_name)
    mode2_1.read(print_formatted=True)  # Read the next struct from the GEB file and print its contents
```

Nesting the `read` method in a loop allows reading in data from GEB data file.

### Simulated data interactive example

We will start with the AGATA multiplicity-30 simulated data that we use for training as an example:

> The AGATA multiplicity-30 simulated data is a dataset comprised of 30 gamma-ray energies (0.08 MeV to 2.6 MeV in 0.09 MeV increments) simulated using GEANT4 (separately and then joined together; there are often gamma-rays that so undetected so the combined multiplicity is less than 30).

```python
import gamma_ray_tracking as gr  # import the package
m30_events, m30_clusters = gr.file_io.load_m30()  # load multiplicity-30 simulated events and clusters (GEANT4; for training)
```

Let us examine the first event in this data:

```python
print(m30_events[0])
```

which has the following output:

```text
<Event 0:
 0: Interaction(x=[0. 0. 0.], e=0.000000, ts=0, crystal_no=0, interaction_type=0)
 1: Interaction(x=[-14.4567  16.1371  10.2848], e=0.080000, ts=0.8, crystal_no=108, seg_no=1, interaction_type=2)
 2: Interaction(x=[-24.0918   5.0444  -5.624 ], e=0.170000, ts=0.842, crystal_no=68, seg_no=15, interaction_type=2)
 3: Interaction(x=[-10.1643  14.6396  16.8034], e=0.136723, ts=0.817, crystal_no=144, seg_no=0, interaction_type=1)
 4: Interaction(x=[-10.0596  14.4962  16.6138], e=0.018628, ts=0.826, crystal_no=144, seg_no=0, interaction_type=1)
 5: Interaction(x=[-10.0977  14.544   16.5205], e=0.104649, ts=0.829, crystal_no=144, seg_no=0, interaction_type=2)
 6: Interaction(x=[ -9.4915 -20.6111  -8.416 ], e=0.101717, ts=0.807, crystal_no=76, seg_no=0, interaction_type=1)
 7: Interaction(x=[-10.1542 -22.1247  -6.6737], e=0.112414, ts=0.887, crystal_no=77, seg_no=11, interaction_type=1)
 8: Interaction(x=[-10.3145 -21.7716  -6.8656], e=0.135870, ts=0.902, crystal_no=77, seg_no=10, interaction_type=2)
 9: Interaction(x=[-11.729  -11.8884 -17.4976], e=0.440000, ts=0.807, crystal_no=34, seg_no=1, interaction_type=2)
10: Interaction(x=[  3.3176 -23.8911  -6.1576], e=0.026095, ts=0.83, crystal_no=79, seg_no=11, interaction_type=1)
...
```

The `Event` contains the `Interaction` objects indicating their position and energy. For this particular simulated data, we also have access to the interaction type (`1` is Compton scattering, `2` is absorption, `3` is pair production, `99` is any interaction following a pair production). We have information about the detector crystal and segment as well (here the detector is set to AGATA in the data loading process, but should be specified for other data; the `load_m30` function exists for convenience for this data).

We can then examine the spectrum of the simulated data:

```python
import matplotlib.pyplot as plt
import numpy as np
energies = []
for event, clusters in zip(m30_events, m30_clusters):
    energies.extend(event.energy_sums(clusters).values())
plt.figure()
plt.hist(energies, bins=np.arange(0,3,0.001), histtype='step')
plt.show()
```

Since this is simulated data, the position and energy resolution are much higher than what would be found in experimental data. To compensate, we artificially add noise to the data by packing together close interactions into a single interaction (typical packing distance is 6 mm) and adding noise to the position and energy in line with an error model ([Siciliano et al. 2021](https://doi.org/10.1140/epja/s10050-021-00385-z) or [Söderström et al. 2011](https://doi.org/10.1016/j.nima.2011.02.089)). Error in position is typically much larger than the error in energy. This overall process is called *packing-and-smearing*.

```python
ps_events, ps_clusters = gr.cluster_tools.pack_and_smear_list(m30_events, m30_clusters)
```

Let's try tracking these events to get some predicted clusters. First we compute clusters using a cone clustering (interactions are grouped by their angular distance from one another with respect to the detector origin):

```python
predicted_clusters - []
for event in ps_events:
    predicted_clusters.append(event.cluster_linkage(alpha_degrees = 10.))  # cluster the interactions together if they are within 10 degrees from one another (with respect to the detector center/target)
```

then we reorder those clusters according to the AFT FOM:

```python
fom_kwargs = {  # the FOM used for tracking
    'fom_method' : 'aft',  # use the AFT FOM
    'start_point' : 0,  # assume gamma-rays are originating from the interaction with index 0 (this is always the detector center/target)
}  # see the documentation of gr.fom_tools.FOM() and gr.fom_tools.single_FOM() for additional information about keyword arguments to the FOM
reordered_clusters = []
for event, pred_clusters in zip(ps_events, predicted_clusters):
    reordered_clusters.append(gr.fom_tools.semi_greedy_clusters(event, pred_clusters, width = 5, stride = 2, **fom_kwargs))  # apply a semi-greedy ordering method to each cluster looking ahead five interactions and accepting the first two at each step
```

If we would like the energies and final FOM from each gamma-ray, we can do the same as before:

```python
new_energies = []
new_foms = []
for event, clusters in zip(ps_events, reordered_clusters):
    new_energies.extend(event.energy_sums(clusters).values())
    new_foms.extend(gr.fom_tools.cluster_FOM(event, clusters=clusters, **fom_kwargs).values())
plt.figure()
plt.hist(new_energies, bins=np.arange(0,3,0.01), histtype='step')
plt.show()

plt.figure()
plt.hist(new_foms, bins=np.arange(0,3,0.01), histtype='step')
plt.show()
```

### Ordering FOM optimization example

FOM features are created using methods in the `fom_tools` and `fom_optimization_data` modules:

```python
import pandas as pd
import gamma_ray_tracking as gr
m30_events, m30_clusters, m30_true_energies = gr.file_io.load_m30(include_energies=True)  # load multiplicity-30 simulated events and clusters (GEANT4; for training); true energies are given by cluster ID
ps_events, ps_clusters = gr.cluster_tools.pack_and_smear_list(m30_events, m30_clusters)  # pack and smear the simulated data
(features, ordered, complete,
energy_sums, lengths, opt_index,
other_index, cluster_ids, acceptable,
event_ids, true_cluster_ids) = gr.fom_optimization_data.make_data(ps_events, ps_clusters, true_energies=m30_true_energies)
feature_names = gr.fom_tools.individual_FOM_feature_names()
features_df = pd.DataFrame(data=features, columns=feature_names)
```

The `features` are FOM elements that ostensibly could be used to order interactions. `ordered` is a boolean vector indicating if the features are associated with the true order (used for ordering model generation). `complete` is a boolean vector indicating if the features are associated with a complete energy deposit (used for suppression model generation). `energy_sums` are the energy sums of the associated cluster, `lengths` are the number of interactions, `opt_index` is the row index for each cluster of the ordered features, `other_index` is the row index of every row (can be replaced with `np.arange(0, len(features))`), `cluster_ids` are the absolute ID of the cluster in the data (not the ID of the source cluster with respect to a single `Event`), `acceptable` is a boolean vector indicating if the first two interactions are in the correct order (all that is really needed for tracking), `event_ids` are the associated `Event` object IDs, `true_cluster_ids` are the cluster IDs of the source cluster (for the multiplicity 30 data, this indicates the true emitted energy).

Given the `features` and `event_ids`, we can train a standard ranking model such as a lambda rank model with XGBoost:

```python
import xgboost as xgb

qid = cluster_ids  # Query ID should be specific to the individual cluster
X = features  # The features of each ordering
y = ~ordered  # We want our ranker to assign a lower value to correctly ordered data
ranker = xgb.XGBRanker(tree_method="hist", objective="pairwise")  # Create a ranking model
ranker.fit(X, y, qid=qid)  # Fit the ranking model
```

Linear ranking models are also useful for their simple implementation and sparsity. In this case, we will be applying the ranking model to get values for an optimization problem, which means we are only interested in the optimal result (the ranking of other values is not important, only relative to the top result). We can then create a linear classification model for clusters of ordering features relative to the true ordering features:

```python
from gamma_ray_tracking.cluster_svm import column_generation
mask = other_index ~= opt_index  # we can safely mask away relative features that are zero (where the data indices do not match)
relative_features = features[other_index] - features[opt_index]
w = column_generation(relative_features[mask], cluster_ids[mask])
foms = np.dot(features, w, axis = 1)
relative_foms = np.dot(relative_features, w, axis = 1)
```

Order optimization methods applied to clusters that have a single relative FOM value that is negative will produce an incorrect order. The goal of the `column_generation` method is to repeatedly search for the features of preferred incorrect orders and attempt to classify them with a relative FOM that is negative (much of the produced data does not need to be used because they do not represent preferred alternatives to the true ordering).

In practice, we need additional normalization of the data to remove `NaN` values and `inf` values before passing it to any ranking model.

## Support

For support, please use the Github issue tracker or [email Thomas Lynn](https://www.anl.gov/staff-directory/contact?to=40Qw.MYNoY0p2).

## Roadmap

Providing adequate documentation is an ongoing effort.

Current efforts are focused on incorporating pair-production tracking into the methods here (and in other tracking codes).

Efforts are also being put towards optimizing the tracking FOM for computational efficiency for deployment in other tracking codes.

Ensemble methods for an ordering FOM are also planned (e.g., energy specific models, or separate models for tracking escaped gamma-rays and complete gamma-rays).

Working with GEANT4 to provide additional training data (e.g., a white noise spectrum to remove any data biases).

Classification models for final FOM assignment (reported in mode1 data files) should also be incorporated directly into the package.

`Coincidence` objects should be preferred method of moving/containing `Interactions` over `Event` objects (which are preferred for tracking).

Additional methods for tracking can be integrated into the tracking methods here. In particular, the graph neural network method of [Andersson and Bäck 2023](https://doi.org/10.1016/j.nima.2022.168000) can be incorporated with other methods such that any deficiencies of one method can be covered by the other.

The extended mode1 datatype is able to store tracked gamma-ray events in a more complete form than the default mode1 format, but does not include a complete representation of the data such that the original event is not completely recreated by loading the data (missing interaction level timestamps, crystal IDs, segment IDs, etc.). The extended mode1 data type is very useful for tracking as it can be used to apply computational Compton suppression after tracking possibly using another method (most of the computation of tracking is spent ordering interactions; suppression is cheap in comparison).

## Authors and acknowledgment

This software is made possible through the efforts of [Thomas Lynn](https://github.com/lynntf) based on initial work by [Dominic Yang](https://orcid.org/0000-0002-9453-2299). Many of the optimization methods (that need some TLC) are authored by Dominic Yang with other contributions by Thomas Lynn and [Sven Leyffer](https://github.com/leyffer/).

The AFT and OFT tracking methods were derived from their respective code bases and Amel Korichi and Torben Lauritsen provided much of the information.

Simulated GEANT4 data packaged with the software was created by Amel Korichi and Torben Lauritsen.

Crystal coordinate matrices for AGATA and GRETA are sourced by Amel Korichi and Torben Lauritsen.

This software was funded by a grant from the United States of America Department of Energy.

## License

This software is licensed using the GNU GPL 2.0 license.

## Copyright

Copyright (C) 2023 Argonne National Laboratory

## Project status

This project is currently in active development.
