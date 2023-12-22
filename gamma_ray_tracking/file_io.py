"""
Copyright (C) 2023 Argonne National Laboratory
This software is provided without warranty and is licensed under the GNU GPL 2.0 license

File in/out for reading/writing GEB files and ASCII simulated files
"""
from __future__ import annotations
import os
import struct
from typing import BinaryIO, Dict, Generator, List, Tuple

import pickle as pkl

import numpy as np
import yaml
from scipy.spatial.distance import pdist, squareform

from . import default_config
from .detector_config_class import DetectorConfig
from .event_class import Event
from .fom_tools import cluster_FOM
from .interaction_class import Interaction


class GEBdata_file:
    """
    # Class to handle reading GEB data files

    A GEB data file is a format for storing gamma-ray tracking data from
    segmented germanium detectors. Each GEB data file consists of a header and a
    list of events that occurred in the detector. Each event consists of one or
    more interactions that have information such as position, energy, time,
    crystal number, segment number, etc.

    This class provides methods to read and parse the GEB data file and return
    event objects that represent the events in the file.
    """

    def __init__(
        self,
        file: BinaryIO = None,
        filename: str = None,
        MAX_INTPTS: int = 16,
        detector: DetectorConfig = default_config,
        global_coords: bool = False,
    ):
        """
        # Initialize the binary formats and configurations

        ## Args:
        - `file` (optional) : A binary file object that contains the GEB data. The
        preferred method is to open the file before instancing the class, using
        `with open(filename, 'rb') as file:`
        - `filename` (optional) : A string that specifies the name of the binary
        file that contains the GEB data. This argument is used only if the file
        object is not provided.
        - `MAX_INTPTS` (optional) : An integer that specifies the maximum number of
        interaction points allowed in an event. The default value is 16.

        ## Attributes:
        - `file` : The binary file object that contains the GEB data
        - `filename` : The name of the binary file that contains the GEB data
        - `MAX_INTPTS` : The maximum number of interaction points allowed in an
        event
        - `GEBHeader_format` : The struct format for parsing the header of the GEB data
        file
        - `intpts_format` : The struct format for parsing an interaction point
        in an event
        - `crystal_intpts_format` : The struct format for parsing information about
        a crystal's recorded interaction points in an event
        - `mode2_format` : The struct format for parsing a MODE2 crystal level event
        - `tracked_gamma_hit_format` : The struct format for parsing the number hits
        recorded in an event MODE1 struct
        - `mode1_format` : The struct format for parsing MODE1 event level data
        """
        if file is None and filename is not None:
            self.file = open(filename, "rb")  # Not the preferred method
        elif file is not None:
            self.file = file
        else:
            raise ValueError("No file provided")
        if isinstance(detector, str):
            detector = DetectorConfig(detector=detector)
        self.crmat = detector.crmat
        self.index = -1
        self.MAX_INTPTS = MAX_INTPTS
        self.GEBHeader_format = "iiq"
        self.intpts_format = "ffffif"
        self.crystal_intpts_format = "iiifiiiiqqfffffffi"
        self.mode2format = (
            self.crystal_intpts_format + self.MAX_INTPTS * self.intpts_format
        )
        self.tracked_gamma_hit_format = "ii"
        self.mode1_format = "fifiqffffffffhhee"
        self.header = {}
        self.global_coords = global_coords

    def read(self, print_formatted: bool = False, as_gr_event: bool = False) -> Dict:
        """
        # Read the next event in the GEB data file

        This method reads the header of the GEB data file to determine the
        format (MODE1 or MODE2) and the data payload. It then parses the data
        payload according to the format and returns an Event object that
        represents the next event in the file.

        ## Args:
        - `print_formatted` (optional) : A boolean that indicates whether to
        print the event information to the standard output in a human-readable
        format. The default value is False, which means the event information is
        not printed.

        ## Returns:
        - An Event object that represents the next event in the GEB data file
        """
        self.index += 1
        data = self.file.read(struct.calcsize(self.GEBHeader_format))
        if not data:
            return None
        data_type, data_length, data_timestamp = struct.unpack(
            self.GEBHeader_format, data
        )
        header = {
            "GEBHEADER type": data_type,
            "GEBHEADER length": data_length,
            "GEBHEADER ts": data_timestamp,
        }
        self.header = header
        if print_formatted:
            print(f"***** event no {self.index}")
            print(f"Header type   {header['GEBHEADER type']}")
            print(f"Header length {header['GEBHEADER length']}")
            print(f"Header ts     {header['GEBHEADER ts']}")
        self.mode2format = (
            self.crystal_intpts_format + self.MAX_INTPTS * self.intpts_format
        )
        if header["GEBHEADER length"] != struct.calcsize(self.mode2format):
            # Data includes extra padding, need to update the data format
            self.mode2format = (
                self.crystal_intpts_format
                + self.MAX_INTPTS * self.intpts_format
                + (header["GEBHEADER length"] - struct.calcsize(self.mode2format)) * "x"
            )
        if data_type == 1:
            try:
                data = self.read_mode2(print_formatted=print_formatted)
            except struct.error:
                return
            # Reset the data format
            self.mode2format = (
                self.crystal_intpts_format + self.MAX_INTPTS * self.intpts_format
            )
        elif data_type == 3:
            try:
                data = self.read_mode1(
                    print_formatted=print_formatted, as_gr_event=as_gr_event
                )
            except struct.error:
                return
            if as_gr_event:
                return data
        elif data_type == 33:
            try:
                data = self.read_extended_mode1(
                    print_formatted=print_formatted, as_gr_event=as_gr_event
                )
            except struct.error:
                return
            if as_gr_event:
                return data
        elif data_type == 11:
            try:
                data = self.read_simulated_data(print_formatted=print_formatted)
            except struct.error:
                return
            return header
        else:
            self.file.read(header["GEBHEADER length"])
            return header
        return header | data

    def read_mode2(self, print_formatted: bool = False, size: int = None) -> Dict:
        """
        # Read in a crystal level event from a MODE2 data file

        This method reads the header of the MODE2 data file to determine the
        data payload size and format. It then parses the data payload and
        returns a dictionary that represents a crystal level event from the
        MODE2 data file.

        ## Args:
        - `print_formatted` (optional) : A boolean that indicates whether to
        print the event information to the standard output in a human-readable
        format. The default value is False, which means the event information is
        not printed.
        - `size` (optional) : An integer that specifies the number of bytes to
        read from the file. The default value is None, which means the default
        data size is read.

        ## Returns:
        A dictionary that represents a crystal level event from the MODE2 data
        file. The dictionary has the following keys and values:
        - `mode` : An integer that indicates the mode of the data. The value is
          2 for MODE2 data.
        - `type` : An integer that as of June 2012 is `abcd5678`
        - `crystal_id` : An integer that indicates id of the reporting crystal
        - `num` : An integer that indicates the number of interactions detected
          in the crystal event
        - `tot_e` : A float that indicates the total energy detected in the
          crystal in keV. This is the central contact (CC) energy for the
          central contact selected for use in decomposition (calibrated, and for
          10 MeV CC channels; includes DNL correction).
        - `core_e` : A list of the four raw core energies from FPGA filter
          (uncalibrated) as ints
        - `timestamp` : An int that indicates the timestamp of the crystal event
          in units (10 ns)
        - `trig_time` : A float that indicates the trigger time of the crystal
          event
        - `t0` : A float that indicates the t0 parameter of the crystal event
        - `cfd` : A float that indicates the cfd parameter of the crystal event
        - `chisq` : A float that indicates the chi-squared value of the pulse
          shape analysis (PSA) for the crystal event
        - `norm_chisq` : A float that indicates the normalized chi-squared value
          of the PSA for the crystal event
        - `baseline` : A float that indicates the baseline value of energy
          traces in the PSA
        - `prestep` : A float that indicates the prestep value of the PSA
        - `poststep` : A float that indicates the poststep value of the PSA
        - `pad` : An integer that indicates the padding bytes in the crystal
          event
        - `intpts` : A list of dictionaries, each representing an interaction
          point in the event. Each dictionary has the following keys and values:
            - `int` : A list of four floats that indicate the position (x, y, z)
              and energy (e) of the interaction point in mm and keV
            - `global_int` : A list of four floats that indicate the global
              position (x, y, z) and energy (e) of the interaction point in mm
              and keV. This is calculated by applying a transformation matrix
              (`crmat`) to the local position based on the crystal id.
            - `seg` : An integer that indicates the segment id of the segment
              reporting the interaction point
            - `seg_ener` : A float that indicates the segment energy of the
              interaction point in keV
        """
        event = {}
        if size is None:
            event_data = struct.unpack(
                self.mode2format, self.file.read(struct.calcsize(self.mode2format))
            )
        else:
            event_data = struct.unpack(self.mode2format, self.file.read(size))
        event["mode"] = 2
        event["type"] = event_data[0]
        event["crystal_id"] = event_data[1]
        event["num"] = event_data[2]
        event["tot_e"] = event_data[3]
        event["core_e"] = event_data[4:8]
        event["timestamp"] = event_data[8]
        event["trig_time"] = event_data[9]
        event["t0"] = event_data[10]
        event["cfd"] = event_data[11]
        event["chisq"] = event_data[12]
        event["norm_chisq"] = event_data[13]
        event["baseline"] = event_data[14]
        event["prestep"] = event_data[15]
        event["poststep"] = event_data[16]
        event["pad"] = event_data[17]

        intpts = []
        # for i in range(self.MAX_INTPTS):
        for i in range(event["num"]):
            intpt = {}
            intpt["int"] = event_data[18 + (i * 6) : 22 + (i * 6)]
            if self.crmat is not None and not self.global_coords:
                try:
                    intpt["global_int"] = np.matmul(
                        self.crmat[event["crystal_id"]],
                        np.array(list(intpt["int"][:3]) + [10]),
                    )
                    intpt["global_int"][3] = intpt["int"][3]
                except ValueError as exc:
                    raise ValueError(
                        f"Problem with coordinate transform. {event} {intpt},"
                        + f" {np.array(list(intpt['int'][:3]) + [10])}"
                    ) from exc
            else:
                intpt["global_int"] = intpt["int"]
            intpt["seg"] = event_data[22 + (i * 6)]
            intpt["seg_ener"] = event_data[23 + (i * 6)]
            intpts.append(intpt)
        event["intpts"] = intpts
        event["sum_e"] = sum(intpt["int"][3] for intpt in event["intpts"])
        corr = event["tot_e"] / event["sum_e"]  # correction factor for energy
        event["energy_factor"] = corr

        if print_formatted:
            print("\n===========================")
            print(
                f"Event at time {event['timestamp']} in crystal {event['crystal_id']}"
            )
            print("num | tot_e (keV) |   t0   | chisq | norm_chisq |   timestamp")
            print("----+-------------+--------+-------+------------+---------------")
            print(
                f"{event['num']:2}  | {event['tot_e']:^11.3f} |"
                + f" {event['t0']:4.3f} | {event['chisq']:^5.3f} |"
                + f" {event['norm_chisq']:^10.3f} | {event['timestamp']}"
            )
            try:
                print("")
                print(
                    "id |  x (cm)  |  y (cm)  |  z (cm)  |   e (keV)  |  seg  | r (cm)"
                )
                print(
                    "---+----------+----------+----------+------------+-------+--------"
                )
                for i, intpt in enumerate(event["intpts"][: event["num"]]):
                    if self.crmat is None:
                        print(
                            f"{i:2} | {intpt['int'][0]/10:8.2f} |"
                            + f" {intpt['int'][1]/10:8.2f} |"
                            + f" {intpt['int'][2]/10:8.2f} |"
                            + f" {intpt['int'][3]*corr:4.0f}/ {event['tot_e']:<4.0f} |"
                            + f" {intpt['seg']:^5} | {np.linalg.norm(intpt['int'][:3])/10:6.2f}"
                        )
                    else:
                        print(
                            f"{i:2} | {intpt['global_int'][0]/10:8.2f} |"
                            + f" {intpt['global_int'][1]/10:8.2f} |"
                            + f" {intpt['global_int'][2]/10:8.2f} |"
                            + f" {intpt['int'][3]*corr:4.0f}/ {event['tot_e']:<4.0f} |"
                            + f" {intpt['seg']:^5} |"
                            + f" {np.linalg.norm(intpt['global_int'][:3])/10:6.2f}"
                        )
            except IndexError:
                print("No interactions read")
            print("")
        return event

    def read_mode1(
        self, print_formatted: bool = False, as_gr_event: bool = False
    ) -> Dict:
        """
        # Read in an event from a MODE1 data file

        This method assumes that the header of the mode1 data file has already
        been read by the `read` method and that the data payload is in MODE1
        format. It then parses the data payload and returns a dictionary that
        represents an event from the MODE1 data file.

        ## Args:
        - `print_formatted` (optional) : A boolean that indicates whether to
        print the event information to the standard output in a human-readable
        format. The default value is False, which means the event information is
        not printed.

        ## Returns:
        A dictionary that represents an event from the MODE1 data file. The
        dictionary has the following keys and values:
        - `mode` : An integer that indicates the mode of the data. The value is
        1 for MODE1 data.
        - `ngam` : An integer that indicates the number of &gamma;-rays in the
        event
        - For each &gamma;-ray detected indexed [0,`ngam`):
            - `pad` : An integer that indicates
                - non-0 with a decomp error, value gives error type
                - `pad = 1`   a null pointer was passed to dl_decomp()
                - `pad = 2`   total energy below threshold
                - `pad = 3`   no net charge segments in evt
                - `pad = 4`   too many net charge segments
                - `pad = 5`   chi^2 is bad following decomp (in this case
                             crys_intpts is non-zero but post-processing
                             step is not applied)
                - `pad = 6`   bad build, i.e. <40 segment+CC channels found
                - `pad|= 128`  PileUp, i.e. pileup flag or deltaT1<6usec
                    - e.g.:
                        - `pad = 128`  pileup+Good
                        - `pad = 133`  pileup+BadChisq
            - `esum` : A float that indicates the energy sum of the &gamma;-ray
              in keV
            - `ndet` : An integer that indicates the number of interactions in
              the detected &gamma;-ray
            - `fom` : A float that indicates the figure of merit of the event
            - `tracked` : An integer that indicates whether the &gamma;-ray was
              tracked or not. The value is 0 for untracked rays and 1 for
              tracked rays.
            - `timestamp` : A float that indicates the timestamp of the
              &gamma;-ray in units (10 ns). This is the timestamp for the first
              interaction in the tracked &gamma;-ray
            - `fhcrID` : An integer that indicates the first hit crystal id of
              the &gamma;-ray
            - `first` : An Interaction object that represents the first
              interaction point of the &gamma;-ray
            - `second` : An Interaction object that represents the second
              interaction point of the &gamma;-ray
            - `escaped` : An integer that indicates whether the &gamma;-ray has
              escaped or not. The value is 0 for no escape and 1 for escape.
            - `TANGO` : A half precision float that indicates the TANGO
              estimated energy of the &gamma;-ray in keV
            - `TANGO_fom` : A float that indicates the figure of merit using the
              TANGO energy of the &gamma;-ray
        """
        ngam, pad = struct.unpack(
            self.tracked_gamma_hit_format,
            self.file.read(struct.calcsize(self.tracked_gamma_hit_format)),
        )

        events = {"mode": 1, "ngam": ngam, "pad": pad}
        for i in range(ngam):
            event = {}
            event_data = tuple(
                struct.unpack(
                    self.mode1_format,
                    self.file.read(struct.calcsize(self.mode1_format)),
                )
            )
            event["esum"] = event_data[0]
            event["ndet"] = event_data[1]
            event["fom"] = event_data[2]
            event["tracked"] = event_data[3]
            event["timestamp"] = event_data[4]
            event["fhcrID"] = event_data[13]
            first_int = tuple(event_data[5:9])
            second_int = tuple(event_data[9:13])
            if as_gr_event:
                first_int = (
                    first_int[0] / 10,
                    first_int[1] / 10,
                    first_int[2] / 10,
                    first_int[3] / 1000,
                )
                second_int = (
                    second_int[0] / 10,
                    second_int[1] / 10,
                    second_int[2] / 10,
                    second_int[3] / 1000,
                )
            event["first"] = Interaction(
                first_int[:3],
                first_int[3],
                ts=event["timestamp"],
                crystal_no=event["fhcrID"],
            )
            event["second"] = Interaction(
                second_int[:3],
                second_int[3],
                ts=event["timestamp"],
                crystal_no=event["fhcrID"],
            )
            # event['first'] = Interaction(event_data[5:8], event_data[8],
            #                              ts=event['timestamp'],
            #                              crystal_no=event['fhcrID'])
            # event['second'] = Interaction(event_data[9:12], event_data[12],
            #                               ts=event['timestamp'],
            #                               crystal_no=event['fhcrID'])
            event["escaped"] = event_data[14]
            event["TANGO"] = event_data[15]
            event["TANGO_fom"] = event_data[16]
            events[i] = event

        if print_formatted:
            print(f"We have {events['ngam']} tracked gamma rays")

            for i in range(events["ngam"]):
                print(f"    [{i}]esum     = {events[i]['esum']}")
                print(f"       fom      = {events[i]['fom']}")
                print(f"       ndet     = {events[i]['ndet']}")
                print(f"       tracked  = {events[i]['tracked']}")
                print(f"       timestamp= {events[i]['timestamp']}")
                print(f"           interaction[0]= {events[i]['first']}")
                if events[i]["ndet"] > 1:
                    print(f"           interaction[1]= {events[i]['second']}")
                print(f"       fhcrID   = {events[i]['fhcrID']}")
                print(f"       escaped  = {events[i]['escaped']}")
                print(f"       TANGO    = {events[i]['TANGO']}")
                print(f"       TANGO_fom= {events[i]['TANGO_fom']}")
            print()
        if as_gr_event:
            points = []
            clusters = {}
            int_count = 0
            ts = events[0]["timestamp"]
            for i in range(events["ngam"]):
                ts = min(ts, events[i]["timestamp"])
                points.append(events[i]["first"])
                if events[i]["ndet"] > 1:
                    points.append(events[i]["second"])
                clusters[i] = list(
                    range(int_count + 1, int_count + 2 + (events[i]["ndet"] > 1))
                )
                int_count += len(clusters[i])
            return Event(ts, points), clusters
        return events

    def read_extended_mode1(
        self, print_formatted: bool = False, as_gr_event: bool = False
    ) -> Dict:
        """
        # Read in a full event from an extended MODE1 data file

        This method assumes that the header of the mode1 data file has already
        been read by the `read` method and that the data payload is in extended
        MODE1 format. It then parses the data payload and returns a dictionary
        that represents an event from the MODE1 data file.

        ## Args:
        - `print_formatted` (optional) : A boolean that indicates whether to
        print the event information to the standard output in a human-readable
        format. The default value is False, which means the event information is
        not printed.

        ## Returns:
        A dictionary that represents an event from the MODE1 data file. The
        dictionary has the following keys and values:
        - `mode` : An integer that indicates the mode of the data. The value is
        1 for MODE1 data.
        - `ngam` : An integer that indicates the number of &gamma;-rays in the
        event
        - For each &gamma;-ray detected indexed [0,`ngam`):
            - `pad` : An integer that indicates
                - non-0 with a decomp error, value gives error type
                - `pad = 1`   a null pointer was passed to dl_decomp()
                - `pad = 2`   total energy below threshold
                - `pad = 3`   no net charge segments in evt
                - `pad = 4`   too many net charge segments
                - `pad = 5`   chi^2 is bad following decomp (in this case
                             crys_intpts is non-zero but post-processing
                             step is not applied)
                - `pad = 6`   bad build, i.e. <40 segment+CC channels found
                - `pad|= 128`  PileUp, i.e. pileup flag or deltaT1<6usec
                    - e.g.:
                        - `pad = 128`  pileup+Good
                        - `pad = 133`  pileup+BadChisq
            - `esum` : A float that indicates the energy sum of the &gamma;-ray
              in keV
            - `ndet` : An integer that indicates the number of interactions in
              the detected &gamma;-ray
            - `fom` : A float that indicates the figure of merit of the event
            - `tracked` : An integer that indicates whether the &gamma;-ray was
              tracked or not. The value is 0 for untracked rays and 1 for
              tracked rays.
            - `timestamp` : A float that indicates the timestamp of the
              &gamma;-ray in units (10 ns). This is the timestamp for the first
              interaction in the tracked &gamma;-ray
            - `fhcrID` : An integer that indicates the first hit crystal id of
              the &gamma;-ray
            - `interactions` : The interactions (there are `ndet` interactions)
              comprising the event
                -
            - `escaped` : An integer that indicates whether the &gamma;-ray has
              escaped or not. The value is 0 for no escape and 1 for escape.
            - `TANGO` : A half precision float that indicates the TANGO
              estimated energy of the &gamma;-ray in keV
            - `TANGO_fom` : A float that indicates the figure of merit using the
              TANGO energy of the &gamma;-ray
        """

        tracked_gamma_hit_format = "ii"  # 8 bytes: ngam and pad
        ray_descriptor_format = "fifiq"  # 24 bytes: esum, ndet, fom, tracked, timestamp
        ray_interaction_format = (
            "ffffihe"  # 24 bytes: xyz e, timestamp_offset, crystal_id, energy_factor
        )
        ray_remaining_format = "hhee"  # 8 bytes: crystal_id, escaped, tango, tango_fom

        ngam, pad = struct.unpack(
            tracked_gamma_hit_format,
            self.file.read(struct.calcsize(tracked_gamma_hit_format)),
        )

        events = {"mode": 1, "ngam": ngam, "pad": pad}
        for i in range(ngam):
            event = {}
            event_descriptor_data = tuple(
                struct.unpack(
                    ray_descriptor_format,
                    self.file.read(struct.calcsize(ray_descriptor_format)),
                )
            )
            event["esum"] = event_descriptor_data[0]
            event["ndet"] = event_descriptor_data[1]
            event["fom"] = event_descriptor_data[2]
            event["tracked"] = event_descriptor_data[3]
            event["timestamp"] = event_descriptor_data[4]

            interactions = []
            event["interactions"] = []
            interaction_data = tuple(
                struct.unpack(
                    ray_interaction_format * event["ndet"],
                    self.file.read(
                        struct.calcsize(ray_interaction_format * event["ndet"])
                    ),
                )
            )
            for j in range(event["ndet"]):
                interactions.append(
                    interaction_data[
                        (len(ray_interaction_format) * j) : (
                            len(ray_interaction_format) * (j + 1)
                        )
                    ]
                )

            remaining_data = tuple(
                struct.unpack(
                    ray_remaining_format,
                    self.file.read(struct.calcsize(ray_remaining_format)),
                )
            )
            event["fhcrID"] = remaining_data[0]
            event["escaped"] = remaining_data[1]
            event["TANGO"] = remaining_data[2]
            event["TANGO_fom"] = remaining_data[3]

            for interaction in interactions:
                if as_gr_event:
                    # Correct units
                    interaction = (
                        interaction[0] / 10,  # mm to cm
                        interaction[1] / 10,  # mm to cm
                        interaction[2] / 10,  # mm to cm
                        interaction[3] / 1000,  # keV to MeV
                        interaction[4],  # timestamp offset
                        interaction[5],  # crystal_id
                        interaction[6],  # energy correction factor
                    )

                event["interactions"].append(
                    Interaction(
                        x=interaction[:3],
                        e=interaction[3],
                        ts=event["timestamp"] + interaction[4],
                        crystal_no=interaction[5],
                        energy_factor=interaction[6],
                    )
                )
            events[i] = event

        if print_formatted:
            print(f"We have {events['ngam']} tracked gamma rays")

            for i in range(events["ngam"]):
                print(f"    [{i}]esum     = {events[i]['esum']}")
                print(f"       fom      = {events[i]['fom']}")
                print(f"       ndet     = {events[i]['ndet']}")
                print(f"       tracked  = {events[i]['tracked']}")
                print(f"       timestamp= {events[i]['timestamp']}")
                for j in range(events[i]["ndet"]):
                    print(
                        f"           interaction[{j}]= {events[i]['interactions'][j]}"
                    )
                print(f"       fhcrID   = {events[i]['fhcrID']}")
                print(f"       escaped  = {events[i]['escaped']}")
                print(f"       TANGO    = {events[i]['TANGO']}")
                print(f"       TANGO_fom= {events[i]['TANGO_fom']}")
            print()
        if as_gr_event:
            points = []
            clusters = {}
            int_count = 0
            ts = events[0]["timestamp"]
            for i in range(events["ngam"]):
                ts = min(ts, events[i]["timestamp"])
                points.extend(events[i]["interactions"])
                clusters[i] = list(
                    range(int_count + 1, int_count + 1 + len(events[i]["interactions"]))
                )
                int_count += len(events[i]["interactions"])
            return Event(ts, points), clusters
        return events

    def read_simulated_data(self, print_formatted: bool = False):
        """
        #define MAX_SIM_GAMMAS 10
        struct g4Sim_abcd1234 {
            int type;
            int num;
            int full;
            struct {
                float e;
                float x, y, z;
                float phi, theta;
                float beta;
            } g4Sim_emittedGamma[MAX_SIM_GAMMAS];
        };
        """
        ray_type, ray_num, ray_full = struct.unpack(
            "iii", self.file.read(struct.calcsize("iii"))
        )
        if print_formatted:
            print(f"  Type {ray_type}; Num {ray_num}; Full {ray_full}")
        format_ = "f" + "iffff" * ray_type
        rays = struct.unpack(format_, self.file.read(struct.calcsize(format_)))
        data = []
        first_value = rays[0]
        if print_formatted:
            print(f"  Some value (beta?) {first_value}")
            print(
                "   unknown val | energy [keV] |   angle 1?   |   angle 2?   |   angle 3?   |"
            )
            print(
                "  -------------------------------------------------------------------------"
            )
        for i in range(ray_type):
            data_dict = {
                "unknown value": rays[5 * i + 1],
                "energy [keV]": rays[5 * i + 2],
                "angle 1?": rays[5 * i + 3],
                "angle 2?": rays[5 * i + 4],
                "angle 3?": rays[5 * i + 5],
            }
            data.append(data_dict)
            if print_formatted:
                print(
                    f"  {rays[5*i + 1]:12} | {rays[5*i + 2]:12.4} | {rays[5*i + 3]:12.4}"
                    + f" | {rays[5*i + 4]:12.4} | {rays[5*i + 5]:12.4} |"
                )
        output = {
            "type (number of interactions?)": ray_type,
            "num (?)": ray_num,
            "full (?)": ray_full,
            "first_value (?)": first_value,
            "rays": data,
        }
        return output

    def close(self):
        """
        Close the GEB data file
        """
        self.file.close()


def mode1_data(
    event: Event,
    clusters: Dict,
    escapes: Dict = None,
    fix_units: bool = True,
    foms: Dict = None,
    monster_size: int = 8,
    include_TANGO: bool = False,
    **FOM_kwargs,
) -> bytes:
    """
    # Create byte MODE1 data for writing

    ## Args:
    - `event` : An Event object that represents the &gamma;-ray event
    - `clusters` : A dictionary that maps cluster ids to lists of Interaction
      objects that belong to the same cluster
    - `escapes` (optional) : A dictionary that maps cluster ids to boolean
      values that indicate whether the cluster represents an escaped &gamma;-ray.
      The default value is None, which means no escapes are detected. This
      argument is not from the original struct specification.
    - `fix_units` (optional) : A boolean that indicates whether to change units
      from [cm] and [MeV] to [mm] and [keV]. The default value is True, which
      means the units are changed from tracking units to the default units for
      MODE1 data.
    - `foms` (optional) : Provide FOM values to avoid computation of the FOM
      again.
    - `monster_size` (optional) : An integer that specifies the maximum number
      of interactions in a cluster to be considered for tracking. The default
      value is 8, which means clusters with more than 8 interactions are not
      tracked.
    - `include_TANGO` (optional) : A boolean that indicates whether to include
      the figure of merit (FOM) calculated using the TANGO estimated energy and
      the energy estimate in the data. The default value is False, which means
      the TANGO energy and corresponding FOM are not included. Extra bytes are
      reserved for one short int (indicating escape) and two half precision
      floats (energy and FOM) (`hee`).
    - `**FOM_kwargs` : Keyword arguments for the FOM function used for
      validation of the tracked &gamma;-rays

    ## Returns:
    - A bytes object that contains the MODE1 data for writing
    """
    GEBHeader_format = "iiq"
    ngam = len(clusters)
    tracked_gamma_hit_format = "ii"
    mode1_format = "fifiqffffffffh"
    extra_data_format = "hee"  # Extra bytes
    total_format = (
        GEBHeader_format
        + tracked_gamma_hit_format
        + ngam * (mode1_format + extra_data_format)
    )

    mode1_output = struct.pack(
        GEBHeader_format,
        int(3.0),
        struct.calcsize(
            tracked_gamma_hit_format + ngam * (mode1_format + extra_data_format)
        ),
        int(event.id),
    )
    mode1_output += struct.pack(tracked_gamma_hit_format, ngam, 0)
    if foms is None:
        foms = cluster_FOM(event, clusters, **FOM_kwargs)
    for i, cluster in clusters.items():
        mode1 = {
            "format": total_format,
            "GEB_type": 3,
            "length": struct.calcsize(tracked_gamma_hit_format + ngam * mode1_format),
            "esum": sum(event.points[i].e for i in cluster),
            "ndet": len(cluster),
            "fom": foms[i],
            "tracked": int(len(cluster) <= monster_size),
            "timestamp": event.points[cluster[0]].ts,
        }
        if escapes is None or not include_TANGO:
            mode1["escaped"] = 0
            mode1["TANGO"] = 0.0
            mode1["TANGO_fom"] = 0.0
        else:
            escape_cluster = event.semi_greedy(
                cluster, width=5, stride=3, estimate_start_energy=True, **FOM_kwargs
            )
            mode1["TANGO"] = event.estimate_start_energy(
                escape_cluster, normalize_by_sigma=False
            )
            if mode1["TANGO"] is None:
                mode1["TANGO"] = 0
                mode1["TANGO_fom"] = 0
            else:
                mode1["TANGO_fom"] = event.FOM(
                    escape_cluster, start_energy=mode1["TANGO"], **FOM_kwargs
                )
            # if escapes[i]:
            #     mode1['escaped'] = 1
            # else:
            #     mode1['escaped'] = 0
            mode1["escaped"] = int(escapes[i])
        # if 0 < mode1['TANGO_fom'] < mode1['fom']:
        #     cluster = escape_cluster
        if fix_units:
            mode1["esum"] *= 1000
            mode1["first"] = list(10 * event.points[cluster[0]].x) + [
                1000 * event.points[cluster[0]].e
            ]
            mode1["second"] = (
                [0, 0, 0, 0]
                if len(cluster) == 1
                else list(10 * event.points[cluster[1]].x)
                + [1000 * event.points[cluster[1]].e]
            )
            mode1["TANGO"] = mode1["TANGO"] * 1000
        else:
            mode1["first"] = list(event.points[cluster[0]].x) + [
                event.points[cluster[0]].e
            ]
            mode1["second"] = (
                [0, 0, 0, 0]
                if len(cluster) == 1
                else list(event.points[cluster[1]].x) + [event.points[cluster[1]].e]
            )
        mode1["fhcrID"] = (
            event.points[cluster[0]].crystal_no
            if event.points[cluster[0]].crystal_no is not None
            else 0
        )

        mode1_output += struct.pack(
            mode1_format + extra_data_format,
            mode1["esum"],
            int(mode1["ndet"]),
            mode1["fom"],
            mode1["tracked"],
            int(mode1["timestamp"]),
            mode1["first"][0],
            mode1["first"][1],
            mode1["first"][2],
            mode1["first"][3],
            mode1["second"][0],
            mode1["second"][1],
            mode1["second"][2],
            mode1["second"][3],
            int(mode1["fhcrID"]),
            mode1["escaped"],
            mode1["TANGO"],
            mode1["TANGO_fom"],
        )
    return mode1_output


def mode1_extended_data(
    event: Event,
    clusters: Dict,
    escapes: Dict = None,
    fix_units: bool = True,
    foms: Dict = None,
    monster_size: int = 8,
    include_TANGO: bool = False,
    **FOM_kwargs,
) -> bytes:
    """
    # Create byte extended MODE1 data for writing

    ## Args:
    - `event` : An Event object that represents the &gamma;-ray event
    - `clusters` : A dictionary that maps cluster ids to lists of Interaction
      objects that belong to the same cluster
    - `escapes` (optional) : A dictionary that maps cluster ids to boolean
      values that indicate whether the cluster represents an escaped &gamma;-ray.
      The default value is None, which means no escapes are detected. This
      argument is not from the original struct specification.
    - `fix_units` (optional) : A boolean that indicates whether to change units
      from [cm] and [MeV] to [mm] and [keV]. The default value is True, which
      means the units are changed from tracking units to the default units for
      MODE1 data.
    - `foms` (optional) : Provide FOM values to avoid computation of the FOM
      again.
    - `monster_size` (optional) : An integer that specifies the maximum number
      of interactions in a cluster to be considered for tracking. The default
      value is 8, which means clusters with more than 8 interactions are not
      tracked.
    - `include_TANGO` (optional) : A boolean that indicates whether to include
      the figure of merit (FOM) calculated using the TANGO estimated energy and
      the energy estimate in the data. The default value is False, which means
      the TANGO energy and corresponding FOM are not included. Extra bytes are
      reserved for one short int (indicating escape) and two half precision
      floats (energy and FOM) (`hee`).
    - `**FOM_kwargs` : Keyword arguments for the FOM function used for
      validation of the tracked &gamma;-rays

    ## Returns:
    - A bytes object that contains the extended MODE1 data for writing
    """
    GEB_type = 33
    GEBHeader_format = "iiq"  # 16 bytes
    ngam = len(clusters)
    tracked_gamma_hit_format = "ii"  # 8 bytes
    gamma_info_format = "fifiq"  # 24 bytes
    interaction_format = "ffffihe"  # 24 bytes
    addendum_format = "hhee"  # 8 bytes
    # mode1_format = 'fifiqffffffffh'
    # extra_data_format = 'hee' # Extra bytes
    total_format = (
        GEBHeader_format
        + tracked_gamma_hit_format
        + ngam * (gamma_info_format + addendum_format)
        + len(event) * interaction_format
    )
    # ngam*(mode1_format + extra_data_format)
    data_format = (
        tracked_gamma_hit_format
        + ngam * (gamma_info_format + addendum_format)
        + len(event) * interaction_format
    )

    mode1_output = struct.pack(
        GEBHeader_format, int(GEB_type), struct.calcsize(data_format), int(event.id)
    )
    pad = 0
    mode1_output += struct.pack(tracked_gamma_hit_format, ngam, pad)
    if foms is None:
        foms = cluster_FOM(event, clusters, **FOM_kwargs)
    for i, cluster in clusters.items():
        mode1 = {
            "format": total_format,
            "GEB_type": GEB_type,
            "length": struct.calcsize(
                tracked_gamma_hit_format
                + ngam * (gamma_info_format + addendum_format)
                + len(event) * interaction_format
            ),
            "esum": sum(event.points[i].e for i in cluster),
            "ndet": len(cluster),
            "fom": foms[i],
            "tracked": int(len(cluster) <= monster_size),
            "timestamp": event.points[cluster[0]].ts,
        }
        if escapes is None or not include_TANGO:
            mode1["escaped"] = 0
            mode1["TANGO"] = 0.0
            mode1["TANGO_fom"] = 0.0
        else:
            escape_cluster = event.semi_greedy(
                cluster, width=5, stride=3, estimate_start_energy=True, **FOM_kwargs
            )
            mode1["TANGO"] = event.estimate_start_energy(
                escape_cluster, normalize_by_sigma=False
            )
            if mode1["TANGO"] is None:
                mode1["TANGO"] = 0
                mode1["TANGO_fom"] = 0
            else:
                mode1["TANGO_fom"] = event.FOM(
                    escape_cluster, start_energy=mode1["TANGO"], **FOM_kwargs
                )
            # if escapes[i]:
            #     mode1['escaped'] = 1
            # else:
            #     mode1['escaped'] = 0
            mode1["escaped"] = int(escapes[i])
        # if 0 < mode1['TANGO_fom'] < mode1['fom']:
        #     cluster = escape_cluster
        if fix_units:
            mode1["esum"] *= 1000
            mode1["interactions"] = []
            for j in cluster:
                mode1["interactions"].append(
                    list(10 * event.points[j].x)
                    + [
                        1000 * event.points[j].e,
                        int(event.points[j].ts - mode1["timestamp"]),
                        int(event.points[j].crystal_no)
                        if event.points[j].crystal_no is not None
                        else 0,
                        event.points[j].energy_factor,
                    ]
                )
            mode1["TANGO"] = mode1["TANGO"] * 1000
        else:
            mode1["interactions"] = []
            for j in cluster:
                mode1["interactions"].append(
                    list(event.points[j].x) + [event.points[j].e]
                )
        mode1["fhcrID"] = (
            event.points[cluster[0]].crystal_no
            if event.points[cluster[0]].crystal_no is not None
            else 0
        )

        mode1_output += struct.pack(
            gamma_info_format,
            mode1["esum"],
            int(mode1["ndet"]),
            mode1["fom"],
            mode1["tracked"],
            int(mode1["timestamp"]),
        )
        for interaction in mode1["interactions"]:
            mode1_output += struct.pack(interaction_format, *interaction)
        mode1_output += struct.pack(
            addendum_format,
            int(mode1["fhcrID"]),
            mode1["escaped"],
            mode1["TANGO"],
            mode1["TANGO_fom"],
        )
    return mode1_output


# TODO - tracked intermediate struct format? Extended mode1 with the full event

# def mode2_loader(file:BinaryIO, time_gap:int=40, debug:bool=False,
#                  print_formatted:bool=False, combine_collisions:bool=True,
#                  monitor_progress:bool=False, coincidence_from:str='first',
#                  buffered:bool=False) -> Generator[Event, None, None]:
#     """
#     # Create a generator for events from a MODE2 file

#     ## Args:
#     - `file` : A binary file object that contains the MODE2 data
#     - `time_gap` : An integer that specifies the maximum time difference (in
#       units [10 ns]) between two consecutive interactions to be considered as
#       part of the same event. The default value is 40 units.
#     - `debug` : A boolean that indicates whether to print debug messages to the
#       standard output. The default value is False.
#     - `print_formatted` : A boolean that indicates whether to print formatted
#       events to the standard output. The default value is False.
#     - `combine_collisions` : A boolean that indicates whether to combine
#       multiple interactions that occurred in the same coordinates into one
#       interaction. The default value is True.
#     - `monitor_progress` : A boolean that indicates whether to return the
#       location in the file while reading. The default value is False.
#     - `coincidence_from` : How to construct the coincidence. The default method
#       is to combine all interactions within the time_gap of the first timestamp
#       of the coincidence.
#     - `buffered` : Use buffered timestamps to handle time travel.

#     ## Returns:
#     - A generator that yields Event objects, each representing an event from the
#       MODE2 file
#     """
#     mode2_data = GEBdata_file(file)
#     coincidence = []
#     old_time = 0
#     struct_reads = 0
#     while True:
#         proto_event = mode2_data.read(print_formatted=print_formatted)
#         struct_reads += 1
#         if proto_event is None: # reached the end
#             event = mode2_coincidence_to_event(coincidence,
#                                                debug=debug,
#                                                combine_collisions=combine_collisions)
#             if event is not None:
#                 if monitor_progress:
#                     yield struct_reads, event
#                 else:
#                     yield event
#             return
#         try:
#             # We assume that timestamps always increase, but this in not always
#             # the case. The data may not be sorted as needed so we use an
#             # absolute value to capture any large deviations in timestamps.
#             if abs(proto_event['timestamp'] - old_time) > time_gap and len(coincidence) > 0:
#                 event = mode2_coincidence_to_event(coincidence,
#                                                    debug=debug,
#                                                    combine_collisions=combine_collisions)
#                 coincidence = []
#                 if event is not None:
#                     if monitor_progress:
#                         yield struct_reads, event
#                     else:
#                         yield event
#             if proto_event['num'] > 0:
#                 coincidence.append(proto_event)
#             if coincidence_from == "first" and len(coincidence) > 0:
#                 old_time = coincidence[0]['timestamp']
#             if coincidence_from != "first":
#                 old_time = proto_event['timestamp']
#         except struct.error:
#             return


def mode2_loader(
    file: BinaryIO,
    time_gap: int = 40,
    debug: bool = False,
    print_formatted: bool = False,
    combine_collisions: bool = True,
    monitor_progress: bool = False,
    coincidence_from: str = "first",
    buffer_size: int = 5,
    detector: DetectorConfig = default_config,
    global_coords: bool = False,
) -> Generator[Event, None, None]:
    """
    # Create a generator for events from a MODE2 file

    ## Args:
    - `file` : A binary file object that contains the MODE2 data
    - `time_gap` : An integer that specifies the maximum time difference (in
      units [10 ns]) between two consecutive interactions to be considered as
      part of the same event. The default value is 40 units.
    - `debug` : A boolean that indicates whether to print debug messages to the
      standard output. The default value is False.
    - `print_formatted` : A boolean that indicates whether to print formatted
      events to the standard output. The default value is False.
    - `combine_collisions` : A boolean that indicates whether to combine
      multiple interactions that occurred in the same coordinates into one
      interaction. The default value is True.
    - `monitor_progress` : A boolean that indicates whether to return the
      location in the file while reading. The default value is False.
    - `coincidence_from` : How to construct the coincidence. The default method
      is to combine all interactions within the time_gap of the first timestamp
      of the coincidence.
    - `buffer_size` : Use buffered timestamps to handle time travel with a
      buffer of some size

    ## Returns:
    - A generator that yields Event objects, each representing an event from the
      MODE2 file
    """
    mode2_data = GEBdata_file(file, detector=detector, global_coords=global_coords)
    buffer = []  # Initialize the buffer to store tuples (timestamp, event)
    coincidence = (
        []
    )  # Initialize the coincidence list to store crystal level events within time_gap
    struct_reads = 0

    # Determine the index from which to find coincidences in the buffer
    if coincidence_from == "first":
        find_index = 0
    else:
        find_index = -1

    while True:
        proto_event = mode2_data.read(print_formatted=print_formatted)
        if proto_event is None:  # Reached the end of the file
            # Yield the last event in the coincidence list (if it exists)
            event = mode2_coincidence_to_event(
                coincidence, debug=debug, combine_collisions=combine_collisions
            )
            if event is not None:
                if monitor_progress:
                    yield struct_reads, event
                else:
                    yield event
            return
        if proto_event["GEBHEADER type"] != 1:
            continue

        buffer.append((proto_event["timestamp"], proto_event))
        struct_reads += 1

        try:
            if len(buffer) == buffer_size:
                if debug:
                    print(f"Buffer size is now {len(buffer)}")
                pop_inds = []  # Store the indices of buffer items to be popped
                if len(coincidence) > 0:
                    if debug:
                        print(
                            "Looking for proto_events to pop from the buffer near"
                            + f" {coincidence[0]['timestamp']}"
                        )
                    for i, buffer_item in enumerate(buffer):
                        if (
                            abs(buffer_item[0] - coincidence[find_index]["timestamp"])
                            < time_gap
                        ):
                            pop_inds.append(i)
                    if debug:
                        print(f"Found pop indices {pop_inds}")
                    for j, i in enumerate(pop_inds):
                        # Pop the relevant buffer items and add them to the coincidence list
                        coincidence.append(buffer.pop(i - j)[1])
                    if debug:
                        print(f"Popped indexes, buffer length now {len(buffer)}")
                    if len(pop_inds) == 0:
                        # If no more items found to pop, yield the event and
                        # reset the coincidence list
                        if debug:
                            print(
                                "Didn't find any more indices to pop, so we can yield an event"
                            )
                        event = mode2_coincidence_to_event(
                            coincidence,
                            debug=debug,
                            combine_collisions=combine_collisions,
                        )
                        coincidence = [buffer.pop(0)[1]]
                        yield event
                else:
                    # If coincidence is empty, just pop the first buffer item
                    # and add it to the coincidence list
                    if debug:
                        print(
                            "Coincidence was empty so we will pop the "
                            + f"current buffer item: {buffer[0]}"
                        )
                    coincidence.append(buffer.pop(0)[1])
        except struct.error:
            return


def mode2_coincidence_to_event(
    coincidence: list[dict], combine_collisions: bool = True, debug: bool = False
) -> Event:
    """
    # Convert mode2 coincidence of crystal proto-events to a single event

    This method converts a list of MODE2 crystal level events (proto-events)
    that belong to the same coincidence window into a single event. A
    coincidence window is a time interval in which two or more &gamma;-rays are
    detected by the detector. A proto-event is a dictionary that represents a
    crystal level event from a MODE2 data file.

    ## Args:
    - `coincidence` : A list of dictionaries, each representing a proto-event
    from a MODE2 data file
    - `combine_collisions` (optional) : A boolean that indicates whether to
    combine multiple interactions that occurred in the same coordinates. The
    default value is True, which means the interactions are combined.
    - (not implemented) `combine_segments` (optional) : A boolean that indicates
    whether to combine multiple interactions that occurred in the same crystal
    and segment into one interaction. The default value is True, which means the
    interactions are combined.
    - `debug` (optional) : A boolean that indicates whether to print debug
    messages to the standard output. The default value is False, which means no
    debug messages are printed.

    ## Returns:
    - An Event object that represents the single event converted from the MODE2
    coincidence

    TODO - Implement segment combination. Combine all interactions in a segment (barycenter)
    """

    def proto_event_hit_points(proto_event):
        return [
            Interaction(
                np.array(intpt["global_int"][:3]) / 10,
                intpt["global_int"][3]
                / 1000
                * proto_event["tot_e"]
                / proto_event["sum_e"],
                ts=proto_event["timestamp"],
                crystal_no=proto_event["crystal_id"],
                seg_no=intpt["seg"],
                crystal_x=np.array(intpt["int"][:3]) / 10,
                energy_factor=proto_event["energy_factor"],
            )
            for intpt in proto_event["intpts"][: proto_event["num"]]
            if intpt["int"][3] > 0
        ]

    hit_points = []
    for proto_event in coincidence:
        hit_points.extend(proto_event_hit_points(proto_event))
    # Sometimes zero energy interactions are included. We need to remove them.
    hit_points = [hit_point for hit_point in hit_points if hit_point.e > 0]
    if len(hit_points) == 0:
        return None
    if combine_collisions:
        try:
            point_matrix = squareform(pdist(np.vstack([p.x for p in hit_points])))
        except:
            print(len(hit_points))
            print(hit_points)
            raise
        m = len(hit_points)
        keep_indices = list(range(m))
        for i in range(m):
            for j in range(i + 1, m):
                if point_matrix[i, j] == 0.0 and j in keep_indices:
                    if debug:
                        print(
                            "Found interactions with the same coordinates"
                            + " up to numerical precision"
                        )
                        print("First")
                        print(hit_points[i])
                        print(hit_points[i].crystal_x)
                        print(hit_points[j])
                        print(hit_points[j].crystal_x)
                    hit_points[i].e += hit_points[j].e
                    keep_indices.remove(j)
        hit_points = [hit_points[i] for i in keep_indices]

    return Event(coincidence[0]["timestamp"], hit_points)


# def read_event_csv(csv_filename: str) -> Event:
#     """
#     Read a single event from csv file
#     """
#     df = pd.read_csv(csv_filename, header=None)
#     N = df.shape[0] # Number of interactions
#     interactions = []
#     for i in range(N):
#         interactions.append(Interaction([df.loc[i,0],df.loc[i,1],
#                                         df.loc[i,2]], df.loc[i,3]))
#     return Event(0, interactions)


def read_agata_simulated_data(
    filename: str, extra_zero: bool = False
) -> Tuple[List[Event], List[Dict]]:
    """
    This function reads in data from an AGATA GEANT4 simulated data file.
    Returns a 2-tuple, first is the list of events and second is the list
    of true clusters of the data.
    """
    with open(filename, "r", encoding="utf-8") as f:
        curr_event = 0
        curr_ray = 0
        curr_step = 1
        curr_interactions = []
        events = []
        all_ray_tracks = []
        ray_tracks = {}
        for line in f:
            if extra_zero:
                event, ray, step, _, energy, x, y, z = line.rstrip().split()
            else:
                event, ray, step, energy, x, y, z = line.rstrip().split()
            event, ray, step = map(int, [event, ray, step])
            energy, x, y, z = map(float, [energy, x, y, z])

            # If at a new event
            if event != curr_event:
                events.append(Event(curr_event, curr_interactions))
                all_ray_tracks.append(ray_tracks)
                curr_interactions = []
                ray_tracks = {}
                curr_event = event
                curr_step = 1
                curr_ray = 0

            # If at a new ray
            if ray != curr_ray:
                ray_tracks[ray] = []
                curr_ray = ray

            ray_tracks[curr_ray].append(curr_step)
            curr_interactions.append(
                Interaction(
                    [x / 10, y / 10, z / 10], energy / 1000, interaction_id=curr_step
                )
            )
            curr_step += 1
        # Add in the last event
        events.append(Event(curr_event, curr_interactions))
        all_ray_tracks.append(ray_tracks)
        return events, all_ray_tracks


def read_gretina_gamma_rays(filename):
    """
    This reads in each of the &gamma;-ray events from a file tracked using
    the ANL method of clustering. See for example data/sven.ascii

    Returns a list of &gamma;-ray Event objects
    """
    with open(filename, "r", encoding="utf-8") as f:
        events = []
        for line in f:
            if line.startswith("filtered"):
                curr_event = []
                event_num = int(line.split("_")[1][:3])
            elif line.startswith("#"):
                # The data are in fields 3..6, ts is at the end
                split_line = line.split()
                split_line = list(map(float, split_line[3:7])) + [
                    int(split_line[-1][3:-1])
                ]
                *x, e, ts = split_line
                curr_event.append(Interaction(x, e, ts=ts))
                # # The data are in fields 3..6
                # *x, e = map(float, line.split()[3:7])
                # curr_event.append(Interaction(x, e))
            elif line.startswith("We"):
                events.append(Event(event_num, curr_event))
    return events


# TODO - align this class and the interaction class
class GammaRay:
    """
    Class for representing single &gamma;-rays
    """

    def __init__(self, energy, origin, theta, phi, beta):
        self.energy = energy
        self.origin = np.array(origin)
        self.theta = theta + np.pi
        self.phi = phi
        self.beta = beta

        x = np.cos(self.theta) * np.sin(self.phi)
        y = np.sin(self.theta) * np.sin(self.phi)
        z = np.cos(self.phi)
        self.pos = np.array([x, y, z])

    def __repr__(self):
        return f"<GammaRay: theta={self.theta}, phi={self.phi}>"


# TODO - replace GammaRay class with Interaction class
def read_simulated_ascii(filename):
    """
    Read UCGretina simulated data.
    """
    with open(filename, "r", encoding="utf-8") as f:

        def get_line():
            line = f.readline().rstrip()
            if line:
                return line

            return None

        def read_E_lines(no_rays):
            rays = []
            for _ in range(no_rays):
                energy, x, y, z, theta, phi, beta = (
                    float(item) for item in get_line().split()
                )
                energy /= 1000
                rays.append(GammaRay(energy, [x, y, z], theta, phi, beta))
            return rays

        def read_event():
            line = get_line()
            if line is None:
                return None

            if line.startswith("E"):
                _, no_rays, full_energy, event_no = line.split()
                no_rays, full_energy, event_no = map(
                    int, [no_rays, full_energy, event_no]
                )
                rays = read_E_lines(no_rays)
                return Event(event_no, []), rays

            D_line = line

            _, no_events, event_no = D_line.split()
            no_events, event_no = int(no_events), int(event_no)
            interactions = []
            for _ in range(no_events):
                C_line = get_line()
                _, crystal_no, n_interactions = C_line.split()
                crystal_no, n_interactions = map(int, [crystal_no, n_interactions])
                for _ in range(n_interactions):
                    int_line = get_line()
                    seg_no, energy, x, y, z = int_line.split()
                    seg_no = int(seg_no)
                    energy, x, y, z = map(float, [energy, x, y, z])
                    interactions.append(
                        Interaction(
                            [x / 10, y / 10, z / 10],
                            energy / 1000,
                            seg_no=seg_no,
                            crystal_no=crystal_no,
                        )
                    )
            next_line = get_line()
            _, no_rays, full_energy, event_no = next_line.split()
            no_rays, full_energy, event_no = map(int, [no_rays, full_energy, event_no])
            rays = read_E_lines(no_rays)
            return Event(event_no, interactions), rays

        events = []
        rays = []
        while True:
            out = read_event()
            if out:
                event, ray = out
                events.append(event)
                rays.append(ray)
            else:
                break
        return events, rays


def load_m30(
    filename: str = "gamma_ray_tracking/data/GammaEvents.Mul30",
    include_energies: bool = False,
) -> Tuple[List, List] | Tuple[List, List, List]:
    """
    # Load the simulated AGATA multiplicity 30 data

    This function reads GEANT4 simulated output with multiplicity 30 format and
    parses the data into Event objects and clusters. Each &gamma;-ray consists of
    one or more interactions that have information such as position, energy,
    time, crystal number, segment number, etc.

    ## Args:
    - `filename` (optional) : A string that specifies the name of the GEANT 4
    output file. The default value is 'data/GammaEvents.Mul30'.
    - `include_energies` (optional) : A boolean that indicates whether to
    include the true energies of the &gamma;-rays in the output. The default value
    is False, which means the true energies are not included.

    ## Returns:
    - A tuple of two or three elements, depending on the value of
    `include_energies`.
        - The first element is a list of Event objects, each representing an
          event from the GEANT4 output file.
        - The second element is a list of dictionaries, each representing
          ordered clusters of interactions that belong to the same &gamma;-ray.
        - The third element (optional) is a list of dictionaries, each
          representing the true energy of a &gamma;-ray.
    """
    events_list = []
    clusters_list = []
    true_energies_list = []
    with open(filename, "r", encoding="utf-8") as file:
        # data_file_path = pathlib.Path(importlib.resources('gamma_ray_tracking', 'data')) / 'GammaEvents.Mul30'
        # with importlib.resources.open_text("gamma_ray_tracking", filename) as file:
        # with open(data_file_path, 'r', encoding="utf-8") as file:
        line = " ".join(file.readline().split()).split()
        while line[0] != "$":
            if line[0] == "GAMMA":
                N = int(line[1])
            line = " ".join(file.readline().split()).split()
        line = " ".join(file.readline().split()).split()
        i = 0
        while line:
            clusters = {}
            true_energies = {}
            interaction_index = 0
            points = []
            for j in range(N):
                if line[0] == "-1":
                    true_energies[j + 1] = float(line[1])
                    clusters[j + 1] = []
                    line = " ".join(file.readline().split()).split()
                    try:
                        while line and line[0] != "-1":
                            interaction_index += 1
                            clusters[j + 1].append(interaction_index)
                            crystal_number = int(line[0])
                            energy = float(line[1]) / 1000
                            position = (
                                float(line[2]) / 10,
                                float(line[3]) / 10,
                                float(line[4]) / 10,
                            )
                            segment = int(line[5])
                            time = float(line[6])
                            int_type = int(line[7])
                            points.append(
                                Interaction(
                                    x=np.array(position),
                                    e=energy,
                                    ts=time,
                                    crystal_no=crystal_number,
                                    seg_no=segment,
                                    interaction_type=int_type,
                                )
                            )
                            line = " ".join(file.readline().split()).split()
                    except IndexError:
                        pass
                    if len(clusters[j + 1]) == 0:
                        del clusters[j + 1]
            if len(points) > 0:
                clusters_list.append(clusters)
                events_list.append(Event(i, points))
                true_energies_list.append(true_energies)
            i += 1
    if include_energies:
        return events_list, clusters_list, true_energies_list
    return events_list, clusters_list


def load_options(options_filename: str = None):
    """Load the tracking options specified by the options filename or the default"""
    # Load default options from the package directory
    package_directory = os.path.dirname(os.path.abspath(__file__))
    default_options_path = os.path.join(package_directory, "track_default.yaml")

    with open(default_options_path, "r", encoding="utf-8") as default_options_file:
        default_options = yaml.safe_load(default_options_file)

    if options_filename is not None:
        with open(options_filename, "r", encoding="utf-8") as options_file:
            loaded_options = yaml.safe_load(options_file)
    else:
        print("Using default options.")
        loaded_options = {}

    # Merge default options with loaded options
    options = {**default_options, **loaded_options}

    return options


def read_filtered(filename):
    """
    Simulated data that looks like filtered data6, data7, data8, data9:
    ```
    event_id 48
    sum 1173.226562 FOM= 0.016966 ---
    0: [1]  342.19965   147.68053 -123.51821 -13.44456
    0: [2]  394.92099   172.64774 -128.68320 -26.28666
    0: [3]    2.49750   176.29221 -144.99864 -28.88522
    0: [4]  117.58041   176.94353 -147.30659 -28.97675
    0: [5]   90.43664   160.73123 -166.50339 -35.12760
    0: [6]  225.59148   159.97557 -166.03114 -37.02897

    event_id 49
    sum 1173.226685 FOM= 0.029693 ---
    0: [1]   78.84626    13.37777 195.61450   7.91038
    0: [2]  810.28137    18.77325 232.19016   0.38992
    0: [3]   18.39922    33.62450 226.86954  27.87122
    0: [4]   23.72584    33.87447 224.53694  33.83388
    0: [5]    9.98914    41.73322 221.58362  40.05963
    0: [6]   31.67374    51.65152 214.15778  58.41708
    0: [7]   48.60516    52.74232 211.87138  58.86965
    0: [8]  151.70592    47.39943 208.37421  61.11026
    sum 1332.498169 FOM= 0.065920 ---
    1: [1]  837.93762   -96.57998  80.42443 -143.45892
    1: [2]  252.28503   -105.62859  74.21603 -145.64665
    1: [3]   69.47177    79.29555 -180.51913  66.46693
    1: [4]   62.90307   102.37600 -189.27686  43.23410
    1: [5]  109.90062   125.91130 -177.46674 120.11867
    ```
    """
    events = []
    clusters = []
    points = []
    current_event = None
    current_clusters = {}
    num_points = 0

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) > 0:
                if parts[0] == "event_id":
                    if current_event is not None:
                        events.append(Event(event_id, points))
                        clusters.append(current_clusters)
                        points = []
                        num_points = 0
                    event_id = int(parts[1])
                    current_clusters = {}
                    current_event = 1

                elif parts[0] == "sum":
                    # energy_sum = float(parts[1])
                    # fom = float(parts[3])
                    continue

                elif parts[0][0].isdigit():
                    cluster_id = int(parts[0][0])
                    point_id = num_points + 1
                    e = float(parts[2]) / 1000.0
                    x = float(parts[3]) / 10.0
                    y = float(parts[4]) / 10.0
                    z = float(parts[5]) / 10.0
                    points.append(Interaction([x, y, z], e))
                    if current_clusters.get(cluster_id) is None:
                        current_clusters[cluster_id] = [point_id]
                    else:
                        current_clusters[cluster_id].append(point_id)
                    num_points += 1

    if current_event is not None:
        events.append(Event(event_id, points))
        clusters.append(current_clusters)

    return events, clusters


def read_GEANT4_raw(filename):
    """
    Data that looks like data7_raw, data8_raw, data9_raw:
    ```
    event_id       1
    ----
    0: [115]  314.27536          -nan   43.42084   36.59559   sum  314.28
    1: [115]  762.83990          -nan   38.13081   42.88185   sum 1077.12
    2: [ 64]  255.65108          -nan  101.90393    8.26922   sum 1332.77

    event_id       2
    ----
    0: [ 52]  626.18518     133.70287  189.45441  -47.76589   sum  626.19
    1: [ 53]   73.42706     158.14946  191.98553  -64.70670   sum  699.61
    2: [ 53]   62.03802     192.79288  180.23076  -77.26333   sum  761.65
    3: [ 49]  317.27740     237.46986  154.87683 -122.68888   sum 1078.93
    5: [ 49]   15.58358     245.66704  153.30949 -132.70291   sum 1094.51
    ```
    """
    events = []
    clusters = []
    points = []
    current_event = None
    current_clusters = {}
    num_points = 0

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if len(parts) > 0:
                if parts[0] == "event_id":
                    if current_event is not None:
                        events.append(Event(event_id, points))
                        clusters.append(current_clusters)
                        points = []
                        num_points = 0
                    event_id = int(parts[1])
                    current_clusters = {}
                    current_event = 1

                elif parts[0] == "----":
                    pass  # Skip separator lines

                elif parts[0][0].isdigit():
                    cluster_id = 0
                    # The crystal number sometimes has a gap between the digit
                    # and the first bracket
                    if len(parts) == 9:
                        crystal_id = int(parts[2][:-1])
                        e = float(parts[3]) / 1000.0
                        x = float(parts[4]) / 10.0
                        y = float(parts[5]) / 10.0
                        z = float(parts[6]) / 10.0
                        points.append(Interaction([x, y, z], e, crystal_no=crystal_id))
                        if current_clusters.get(cluster_id) is None:
                            current_clusters[cluster_id] = [num_points + 1]
                        else:
                            current_clusters[cluster_id].append(num_points + 1)
                        num_points += 1
                    elif len(parts) == 8:
                        crystal_id = int(parts[1][1:-1])
                        e = float(parts[2]) / 1000.0
                        x = float(parts[3]) / 10.0
                        y = float(parts[4]) / 10.0
                        z = float(parts[5]) / 10.0
                        points.append(Interaction([x, y, z], e, crystal_no=crystal_id))
                        if current_clusters.get(cluster_id) is None:
                            current_clusters[cluster_id] = [num_points + 1]
                        else:
                            current_clusters[cluster_id].append(num_points + 1)
                        num_points += 1

    if current_event is not None:
        events.append(Event(event_id, points))
        clusters.append(current_clusters)

    return events, clusters


# %% Intermediate tracked format saving and loading: not mode2, not mode1

# It is useful to have an intermediate format where we can save the tracked event
# and clusters without having throw away any data (as in mode1)


def write_events_clusters(file: BinaryIO, events: list[Event], clusters: list[dict]):
    """
    Write a list of events and clusters to an intermediate datatype
    containing the full event and clusters for a tracked event
    """
    for event, clustering in zip(events, clusters):
        write_event_cluster(file, event, clustering)


def write_event_cluster(file: BinaryIO, event: Event, clustering: dict):
    """
    Write the event and clustering to the file
    """
    pkl.dump((event.coincidence, clustering), file)


def read_event_cluster(
    file: BinaryIO, detector_configuration: DetectorConfig = None
) -> tuple[Event, dict]:
    """
    Read the event object and its clustering
    """
    coincidence, clustering = pkl.load(file)
    return (
        coincidence.to_event(detector_configuration=detector_configuration),
        clustering,
    )


def tracked_generator(file: BinaryIO, detector_configuration: DetectorConfig = None):
    """
    Read the file in pieces via a generator
    """
    while True:
        try:
            ev, clu = read_event_cluster(
                file, detector_configuration=detector_configuration
            )
            detector_configuration = ev.detector_config
            yield ev, clu
        except EOFError:
            break


def tracked_generator_filename(
    filename: str, detector_configuration: DetectorConfig = None
):
    """
    Read the file in pieces via a generator
    """
    with open(filename, "rb") as file:
        while True:
            try:
                ev, clu = read_event_cluster(
                    file, detector_configuration=detector_configuration
                )
                detector_configuration = ev.detector_config
                yield ev, clu
            except EOFError:
                break


def read_events_clusters(
    file: BinaryIO, detector_configuration: DetectorConfig = None
) -> tuple[list[Event], list[dict]]:
    """
    Read multiple events and their clusterings
    """
    events = []
    clusters = []
    while True:
        try:
            event, clustering = read_event_cluster(
                file, detector_configuration=detector_configuration
            )
            events.append(event)
            clusters.append(clustering)
            detector_configuration = event.detector_config
        except EOFError:
            return events, clusters
