"""
Copyright (C) 2024 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

Class for storing just the basics for a gamma-ray coincidence of interactions
"""
from dataclasses import dataclass
from typing import List

from greto.detector_config_class import DetectorConfig
from greto.interaction_class import Interaction


@dataclass
class Coincidence:
    """
    A bare-bones class for storing the information that makes up an event

    - event_id: identifier for the event
    - points: list of interaction points
    - detector_name: string name of the detector
    """

    event_id: int
    points: List[Interaction]
    detector: str

    def to_event(self, detector_configuration: DetectorConfig = None):
        """
        Convert the coincidence to an Event

        - detector_configuration: the DetectorConfig class that defines the
          detector. We would like this to be read in so that it does not need to
          be created anew (involves reading data files)

        Returns
        - Event: the full gamma-ray event for calculating FOMs, etc.
        """
        from greto.event_class import Event

        if detector_configuration is not None:
            return Event(self.event_id, self.points, detector=detector_configuration)
        return Event(self.event_id, self.points, detector=self.detector)
