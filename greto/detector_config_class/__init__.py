"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Detector configuration class
"""
from __future__ import annotations

import importlib.resources
import json
import struct

import numpy as np

from greto.interaction_class import Interaction


def read_greta_crmat(filename):
    """
    Read crmat.LINUX file to get rotation matrices for crystal coordinates to
    real coordinates

    Data is arranged:
    [[xx, xy, xz, TrX],
     [yx, yy ,yz, TrY],
     [zx, zy, zz, TrZ],
     [0,  0,  0,  1  ]]
    Translation from crystal to world coordinates is:
    np.matmul(crmat['crystal_id'], [px, py, pz, 1.])
    """
    if filename is None:
        return None
    with importlib.resources.open_binary(__name__, filename) as file:
        crmat = np.array(
            struct.unpack("f" * 16 * 124, file.read(struct.calcsize("f" * 16 * 124)))
        ).reshape((124, 4, 4))
    return crmat


def read_agata_crmat(filename):
    """
    Read GANIL_AGATA_crmat.dat file to get rotation matrices for crystal
    coordinates to real coordinates

    Data is arranged:
    [[TrX, TrY, TrZ],
     [xx,  xy,  xz ],
     [yx,  yy,  yz ],
     [zx,  zy,  zz ]]

    Data should be translated to match the GRETINA crmat:
    [[xx, xy, xz, TrX],
     [yx, yy ,yz, TrY],
     [zx, zy, zz, TrZ],
     [0,  0,  0,  1  ]]
    """
    if filename is None:
        return None
    with importlib.resources.open_text(__name__, filename) as file:
        crmat = []
        for _ in range(180 * 4):
            line = file.readline().split()
            if len(line) == 5:
                crmat.append([float(value) for value in line[2:] + [0.0]])
            if len(line) == 4:
                crmat[-1].extend([float(value) for value in line[1:] + [0.0]])
        crmat = np.array(crmat).reshape((180, 4, 4))
        crmat[:, [0, 1, 2, 3], :] = crmat[:, [1, 2, 3, 0], :]
        for i in range(4):
            crmat[:, i, 3] = crmat[:, 3, i]
            crmat[:, 3, i] = 0.0
        crmat[:, 3, 3] = 10.0
        crmat[:, :, 3] /= 10.0  # Convert units
    return crmat


class DetectorConfig:
    """
    Load detector configuration
    """

    def __init__(
        self,
        detector: str = "agata",
        config_file="detector_config.json",
        position_error_model: str = "siciliano",
    ):
        with importlib.resources.open_text(
            __name__, config_file, encoding="utf-8"
        ) as f:
            config = json.load(f)
        self.agata_inner_radius = config["AGATA_INNER_RADIUS_cm"]
        self.agata_outer_radius = config["AGATA_OUTER_RADIUS_cm"]
        self.greta_inner_radius = config["GRETA_INNER_RADIUS_cm"]
        self.greta_outer_radius = config["GRETA_OUTER_RADIUS_cm"]
        self.inner_radius = 0.0
        self.outer_radius = 0.0
        self.agata_crmat_filename = config["AGATA_CRMAT"]
        self.greta_crmat_filename = config["GRETA_CRMAT"]
        self.greta_crmat = read_greta_crmat(self.greta_crmat_filename)
        self.agata_crmat = read_agata_crmat(self.agata_crmat_filename)
        self.crmat = self.agata_crmat
        self.position_error_w0 = config["POSITION_ERROR_W0_cm"]
        self.position_error_w1 = config["POSITION_ERROR_W1_cm"]
        self.position_error_w0_agata = config["POSITION_ERROR_W0_cm_agata"]
        self.position_error_w1_agata = config["POSITION_ERROR_W1_cm_agata"]
        self.position_error_w0_Soederstroem = config[
            "POSITION_ERROR_W0_cm_Soederstroem"
        ]
        self.position_error_w1_Soederstroem = config[
            "POSITION_ERROR_W1_cm_Soederstroem"
        ]
        self.position_error_w0_Siciliano = config["POSITION_ERROR_W0_cm_Siciliano"]
        self.position_error_w1_Siciliano = config["POSITION_ERROR_W1_cm_Siciliano"]
        self.position_error_model = position_error_model
        self.set_position_uncertainty(self.position_error_model)
        self.origin = Interaction(
            [
                config["default_origin_x_cm"],
                config["default_origin_y_cm"],
                config["default_origin_z_cm"],
            ],
            0.0,
            crystal_no=0,
            crystal_x=[
                config["default_origin_x_cm"],
                config["default_origin_y_cm"],
                config["default_origin_z_cm"],
            ],
            interaction_type=0,
        )
        self.detector = detector
        self.set_detector(self.detector)

    def __repr__(self):
        return f"DetectorConfig(detector={self.detector})"

    def __str__(self) -> str:
        s = "<DetectorConfig:"
        s += f"detector={self.detector}, "
        s += f"inner_radius={self.inner_radius}, "
        s += f"outer_radius={self.outer_radius}, "
        s += f"position_error_model={self.position_error_model}, "
        s += f"position_error_w0={self.position_error_w0}, "
        s += f"position_error_w1={self.position_error_w1}, "
        s += f"origin={self.origin}"
        s += ">"
        return s

    def get_inner_radius(self) -> float:
        """
        Getter for inner radius
        """
        return self.inner_radius

    def set_radius(self, inner: float = 0.0, outer: float = 0.0):
        """
        Set the detector radii. Only the inner radius is required for most calculations.
        """
        self.inner_radius = inner
        self.outer_radius = outer

    def set_detector(self, detector: str):
        """
        Set the detector type to either AGATA or GRETA/GRETINA.
        """
        if detector.lower().startswith("agata"):
            self.set_radius(
                inner=self.agata_inner_radius, outer=self.agata_outer_radius
            )
            self.crmat = self.agata_crmat
            self.detector = "AGATA"
        elif detector.lower().startswith("gret"):
            self.set_radius(
                inner=self.greta_inner_radius, outer=self.greta_outer_radius
            )
            self.crmat = self.greta_crmat
            self.detector = "GRETINA"
        elif detector is None:
            self.crmat = None
            self.detector = None
        else:
            self.set_detector("agata")
            raise ValueError("Invalid detector type, falling back to AGATA")

    def set_position_uncertainty(self, method: str):
        """
        Set the position uncertainty parameters to either AGATA default,
        Soederstroem, or Siciliano.
        """
        # match method.lower():
        #     case "agata":
        #         self.position_error_w0 = self.position_error_w0_agata
        #         self.position_error_w1 = self.position_error_w1_agata
        #     case "soederstroem":
        #         self.position_error_w0 = self.position_error_w0_Soederstroem
        #         self.position_error_w1 = self.position_error_w1_Soederstroem
        #     case "siciliano" | "default":
        #         self.position_error_w0 = self.position_error_w0_Siciliano
        #         self.position_error_w1 = self.position_error_w1_Siciliano
        #     case _:
        #         self.set_position_uncertainty("default")
        #         raise ValueError(
        #             "Invalid uncertainty method, falling back to default Siciliano"
        #         )

        if method.lower() == "agata":
            self.position_error_w0 = self.position_error_w0_agata
            self.position_error_w1 = self.position_error_w1_agata
        elif method.lower() == "soederstroem":
            self.position_error_w0 = self.position_error_w0_Soederstroem
            self.position_error_w1 = self.position_error_w1_Soederstroem
        elif method.lower() == "siciliano" or method.lower() == "default":
            self.position_error_w0 = self.position_error_w0_Siciliano
            self.position_error_w1 = self.position_error_w1_Siciliano
        else:
            self.set_position_uncertainty("default")
            raise ValueError('Invalid uncertainty method, falling back to default Siciliano')

    def position_error(self, energy: np.ndarray):
        """
        Return the expected value of the error in position [cm] as a function of
        deposited energy [MeV].
        """
        out = self.position_error_w0 * np.ones(energy.shape)
        out[energy > 0] = out[energy > 0] + self.position_error_w1 * np.sqrt(
            0.100 / energy[energy > 0]
        )
        return out


default_config = DetectorConfig()
default_config.set_detector("agata")
