"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

# Package for processing &gamma;-ray interactions detected in AGATA or GRETINA
"""
__version__ = "1.0.0"

# %% Imports

from scipy.constants import physical_constants

from gamma_ray_tracking.detector_config_class import default_config
from gamma_ray_tracking.event_class import Event
from gamma_ray_tracking.interaction_class import Interaction
import gamma_ray_tracking.file_io as file_io

# %% Define constants
RHO_GE = 5.323  # Density of Germanium [g/cm3] (NIST)
Z_GE = 32  # Atomic Number of Germanium [p+/atom]
A_GE = 74  # Atomic Mass of Germanium [g/mol]
Z_A_GE = 0.44071  # Ratio of Z/A of Germanium (NIST)
I_GE = 350.0  # Mean excitation energy of Germanium [eV] (NIST)
N_AV = physical_constants["Avogadro constant"][0]  # Avogadro's number [atoms/mol]
R_0 = (
    physical_constants["classical electron radius"][0] * 100
)  # Classical electron Radius [cm]
MEC2 = physical_constants["electron mass energy equivalent in MeV"][
    0
]  # electron mass [MeV]
ALPHA = physical_constants["fine-structure constant"][
    0
]  # fine structure constant 1/137

# %% Default event and clusters for testing (non-physical)
default_event = Event(
    0,
    [
        Interaction([25, 0, 0], 0.1, ts=0, crystal_no=0, seg_no=0, energy_factor=1.0),
        Interaction([25, 4, 0], 0.2, ts=0, crystal_no=1, seg_no=0, energy_factor=1.1),
        Interaction([25, 4, 3], 0.3),
        Interaction([0, 24, 12], 0.4),
        Interaction([5, 24, 0], 0.5),
        Interaction([0, 24, 0], 0.6),
        Interaction([16, 16, 16], 0.7),
        Interaction([0, 0, 25], 0.8),
        Interaction([-16, -16, -16], 0.9),
    ],
)
default_clusters = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7], 4: [8, 9]}
