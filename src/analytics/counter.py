"""
src/analytics/counter.py
========================
Traffic counting module.

Maintains a set of *unique* tracking IDs seen across the whole video so
each object is counted exactly once regardless of how many frames it appears.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


class TrafficCounter:
    """
    Count unique tracked objects by class across a video.

    Usage
    -----
    counter = TrafficCounter()
    counter.update(track_id=12, class_id=2)   # saw car #12
    counter.update(track_id=12, class_id=2)   # same car – ignored
    counts = counter.get_counts()
    """

    def __init__(self):
        # Maps class_id → set of unique tracking IDs seen so far
        self._seen: dict[int, set] = {}

    def update(self, track_id: int, class_id: int) -> bool:
        """
        Register a detection.

        Parameters
        ----------
        track_id : int
            ByteTrack ID assigned to the object.
        class_id : int
            COCO class ID (0=person, 2=car, …).

        Returns
        -------
        bool
            True if this is a *new* object (first time we see this track_id
            for this class), False if it was already counted.
        """
        if class_id not in self._seen:
            self._seen[class_id] = set()
        if track_id in self._seen[class_id]:
            return False
        self._seen[class_id].add(track_id)
        return True

    def get_counts(self) -> dict:
        """
        Return a dictionary with counts for each tracked class plus totals.

        Example return value::

            {
                "total_vehicles": 34,
                "cars": 20,
                "motorcycles": 8,
                "buses": 2,
                "trucks": 4,
                "bicycles": 0,
                "pedestrians": 10,
                "traffic_lights": 1,
            }
        """
        def _count(class_id: int) -> int:
            return len(self._seen.get(class_id, set()))

        vehicles = sum(_count(cid) for cid in config.VEHICLE_CLASS_IDS)

        return {
            "total_vehicles": vehicles,
            "cars": _count(2),
            "motorcycles": _count(3),
            "buses": _count(5),
            "trucks": _count(7),
            "bicycles": _count(1),
            "pedestrians": _count(0),
            "traffic_lights": _count(9),
        }

    def reset(self):
        """Clear all counts (useful between videos)."""
        self._seen.clear()

