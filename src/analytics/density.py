"""
src/analytics/density.py
========================
Traffic-density estimation module.

Density is estimated per-frame based on the number of *vehicles currently
visible* in that frame (not the cumulative count).  This gives a real-time
view of how congested the road appears at any moment.

Thresholds are read from config.py and are therefore easy to adjust.

Density levels
--------------
LOW    : 0 – DENSITY_LOW_THRESHOLD  vehicles in frame
MEDIUM : DENSITY_LOW_THRESHOLD+1 – DENSITY_HIGH_THRESHOLD
HIGH   : > DENSITY_HIGH_THRESHOLD
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def calculate_density(vehicle_count_in_frame: int) -> str:
    """
    Map the number of vehicles visible in a single frame to a density label.

    Parameters
    ----------
    vehicle_count_in_frame : int
        Number of vehicle detections in the current frame
        (i.e. from classes in config.VEHICLE_CLASS_IDS).

    Returns
    -------
    str
        "LOW", "MEDIUM", or "HIGH"
    """
    if vehicle_count_in_frame <= config.DENSITY_LOW_THRESHOLD:
        return "LOW"
    elif vehicle_count_in_frame <= config.DENSITY_HIGH_THRESHOLD:
        return "MEDIUM"
    else:
        return "HIGH"


def density_color_bgr(density: str) -> tuple:
    """Return a BGR colour suitable for overlaying the density label."""
    return {
        "LOW":    (0, 200, 0),      # green
        "MEDIUM": (0, 165, 255),    # orange
        "HIGH":   (0, 0, 220),      # red
    }.get(density, (200, 200, 200))


class DensityTracker:
    """
    Keep a rolling history of density labels (one per sampled frame).
    Useful for the time-series chart.
    """

    def __init__(self):
        self.history: list[dict] = []  # [{"frame": int, "density": str, "vehicle_count": int}]

    def record(self, frame_index: int, vehicle_count_in_frame: int):
        """Record density for a given frame."""
        density = calculate_density(vehicle_count_in_frame)
        self.history.append({
            "frame": frame_index,
            "density": density,
            "vehicle_count": vehicle_count_in_frame,
        })

    def get_overall_density(self) -> str:
        """
        Return the *most common* density label across all recorded frames.
        Falls back to "LOW" if no data.
        """
        if not self.history:
            return "LOW"
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for entry in self.history:
            counts[entry["density"]] += 1
        return max(counts, key=counts.get)

    def to_list(self) -> list[dict]:
        """Return the raw history list for serialisation."""
        return self.history

