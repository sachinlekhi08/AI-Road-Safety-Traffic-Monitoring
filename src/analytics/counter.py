"""
src/analytics/counter.py
========================
Traffic counting module.

Maintains unique tracked objects and separately records vehicles
that cross the virtual counting line.

The same tracking ID is counted only once.
Class statistics are determined from the observed class history
of each tracked object to reduce frame-to-frame class switching.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import config


class TrafficCounter:
    """
    Track unique detected objects and line-crossing vehicles.

    Detection statistics:
        Count every unique tracked object seen in the video.

    Flow statistics:
        Count a vehicle when its tracking ID crosses the virtual line.
    """

    def __init__(self):
        # track_id -> {class_id: number_of_observations}
        self._class_votes: dict[int, dict[int, int]] = {}

        # Tracking IDs that crossed the virtual counting line.
        self._crossed_ids: set[int] = set()

    # ------------------------------------------------------------------
    # Detection counting
    # ------------------------------------------------------------------

    def update(self, track_id: int, class_id: int) -> bool:
        """
        Register an observation of a tracked object.

        Returns True only the first time this tracking ID is observed.
        """

        if track_id not in self._class_votes:
            self._class_votes[track_id] = {}

        votes = self._class_votes[track_id]

        votes[class_id] = votes.get(class_id, 0) + 1

        return sum(votes.values()) == 1

    # ------------------------------------------------------------------
    # Stable class
    # ------------------------------------------------------------------

    def get_stable_class(self, track_id: int) -> int | None:
        """
        Return the most frequently observed class for a tracking ID.
        """

        votes = self._class_votes.get(track_id)

        if not votes:
            return None

        return max(votes, key=votes.get)

    # ------------------------------------------------------------------
    # Line crossing
    # ------------------------------------------------------------------

    def register_crossing(self, track_id: int) -> bool:
        """
        Register a vehicle crossing the virtual line.

        Returns True only the first time that tracking ID crosses.
        """

        if track_id in self._crossed_ids:
            return False

        self._crossed_ids.add(track_id)

        return True

    # ------------------------------------------------------------------
    # Count helpers
    # ------------------------------------------------------------------

    def _count_tracks(self, track_ids=None) -> dict:
        """
        Count objects by their stable class.

        If track_ids is supplied, only those IDs are counted.
        """

        if track_ids is None:
            track_ids = self._class_votes.keys()

        counts = {
            0: 0,  # person
            1: 0,  # bicycle
            2: 0,  # car
            3: 0,  # motorcycle
            5: 0,  # bus
            7: 0,  # truck
            9: 0,  # traffic light
        }

        for track_id in track_ids:

            class_id = self.get_stable_class(track_id)

            if class_id in counts:
                counts[class_id] += 1

        return counts

    # ------------------------------------------------------------------
    # Public statistics
    # ------------------------------------------------------------------

    def get_counts(self) -> dict:
        """
        Return unique detection statistics.
        """

        counts = self._count_tracks()

        vehicles = sum(
            counts.get(cid, 0)
            for cid in config.VEHICLE_CLASS_IDS
        )

        return {
            "total_vehicles": vehicles,
            "cars": counts.get(2, 0),
            "motorcycles": counts.get(3, 0),
            "buses": counts.get(5, 0),
            "trucks": counts.get(7, 0),
            "bicycles": counts.get(1, 0),
            "pedestrians": counts.get(0, 0),
            "traffic_lights": counts.get(9, 0),
        }

    def get_crossing_counts(self) -> dict:
        """
        Return traffic-flow statistics for objects that crossed
        the virtual counting line.
        """

        counts = self._count_tracks(self._crossed_ids)

        vehicles = sum(
            counts.get(cid, 0)
            for cid in config.VEHICLE_CLASS_IDS
        )

        return {
            "crossed_vehicles": vehicles,
            "crossed_cars": counts.get(2, 0),
            "crossed_motorcycles": counts.get(3, 0),
            "crossed_buses": counts.get(5, 0),
            "crossed_trucks": counts.get(7, 0),
            "crossed_bicycles": counts.get(1, 0),
            "crossed_pedestrians": counts.get(0, 0),
        }

    def reset(self):
        """Clear all statistics."""

        self._class_votes.clear()
        self._crossed_ids.clear()