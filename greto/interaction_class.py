"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Gamma-ray Interaction class
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import numpy as np


@dataclass
class Interaction:
    """
    A class representing a single &gamma;-ray interaction

    Args:
        - x: A point in R3 where the interaction occurred
        - e: The energy deposited
        - ts: The time the interaction occurred
        - crystal_no: The crystal id number
        - seg_no: The segment id number
        - event_no: The event id number
        - interaction_id: An identifier for the interaction
        - interaction_type: The type of interaction:
            - `0` - source
            - `1` - Compton scattering
            - `2` - absorption
            - `3` - pair production
            - `99` - interaction after a pair production or possibly other interaction
        - energy_factor: The correction factor used to adjust energy to match
            the central contact (exponential); (central contact energy) / (energy sum)
    """

    x: Iterable[float]
    e: float
    ts: int = 0
    crystal_no: int = None
    seg_no: int = None
    crystal_x: Iterable[float] = None
    event_no: int = None
    interaction_id: str = None
    interaction_type: int = None
    energy_factor: float = None
    pad: int = 0

    def __post_init__(self):
        """Convert to numpy arrays"""
        self.x = np.array(self.x, dtype=float)
        self.crystal_x = (
            np.array(self.crystal_x) if self.crystal_x is not None else None
        )

    @property
    def id(self):
        """Interaction identifier"""
        return self.interaction_id

    @cached_property
    def r(self) -> float:
        """Distance from target"""
        # return np.linalg.norm(self.x)
        return np.sqrt(np.sum(self.x**2))

    @cached_property
    def theta(self) -> float:
        """Angular position (spherical)"""
        if self.r == 0:
            return 0.0
        return np.arccos(self.x[2] / self.r)

    @cached_property
    def phi(self) -> float:
        """Angular position (spherical)"""
        if self.r == 0:
            return 0.0
        return np.arctan2(self.x[1], self.x[0])

    def __str__(self):
        out = f"Interaction(x={self.x}, e={self.e:.6f}, ts={self.ts}"
        if self.crystal_no is not None:
            out += f", crystal_no={self.crystal_no}"
        if self.seg_no is not None:
            out += f", seg_no={self.seg_no}"
        if self.interaction_type is not None:
            out += f", interaction_type={self.interaction_type}"
        if self.energy_factor is not None:
            out += f", energy_factor={self.energy_factor}"
        out += f", pad={self.pad}"
        out += ")"
        return out

    def update_position(self, new_position: Iterable):
        """
        Create a new Interaction instance with updated spatial position.

        Args:
        - `new_position`: A new spatial position represented as an iterable of
        floats.

        Returns:
        A new Interaction instance with the same attributes as the current
        instance except for the updated position.
        """
        updated_instance = Interaction(
            x=new_position,
            e=self.e,
            ts=self.ts,
            crystal_no=self.crystal_no,
            seg_no=self.seg_no,
            crystal_x=self.crystal_x,
            event_no=self.event_no,
            interaction_id=self.interaction_id,
            interaction_type=self.interaction_type,
            energy_factor=self.energy_factor,
        )
        return updated_instance
