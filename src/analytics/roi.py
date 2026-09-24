"""
src/analytics/roi.py
====================
Region-of-Interest (ROI) system.

An ROI is a named polygon (or rectangle) drawn on the video frame.
Objects whose bounding-box centre falls inside an ROI are flagged.

This module is intentionally simple and modular so that different videos
(with different camera angles) can define their own ROI layouts via a
plain dict or JSON without changing any code.

Example ROI config
------------------
roi_config = {
    "zebra_crossing": [(320, 400), (640, 400), (640, 500), (320, 500)],
    "intersection":   [(100, 100), (700, 100), (700, 600), (100, 600)],
}
"""

import cv2
import numpy as np


class ROIManager:
    """
    Manages a set of named ROI polygons and checks whether points / boxes
    fall inside them.
    """

    def __init__(self, roi_config: dict | None = None):
        """
        Parameters
        ----------
        roi_config : dict
            Maps ROI name (str) → list of (x, y) vertex tuples.
            Example::

                {
                    "zebra_crossing": [(100, 400), (500, 400),
                                       (500, 500), (100, 500)],
                }
        """
        self.rois: dict[str, np.ndarray] = {}
        if roi_config:
            for name, points in roi_config.items():
                self.add_roi(name, points)

    # ------------------------------------------------------------------
    # ROI registration
    # ------------------------------------------------------------------

    def add_roi(self, name: str, points: list):
        """Add or replace a named ROI polygon."""
        pts = np.array(points, dtype=np.int32)
        self.rois[name] = pts

    def remove_roi(self, name: str):
        """Remove a named ROI (no-op if not found)."""
        self.rois.pop(name, None)

    def list_rois(self) -> list[str]:
        """Return names of all defined ROIs."""
        return list(self.rois.keys())

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def point_in_roi(self, x: int, y: int, roi_name: str) -> bool:
        """
        Return True if point (x, y) is inside the named ROI polygon.
        Uses OpenCV pointPolygonTest (positive = inside).
        """
        if roi_name not in self.rois:
            return False
        result = cv2.pointPolygonTest(self.rois[roi_name], (float(x), float(y)), False)
        return result >= 0

    def box_centre_in_roi(
        self,
        x1: int, y1: int, x2: int, y2: int,
        roi_name: str,
    ) -> bool:
        """
        Return True if the *centre* of bounding box [x1,y1,x2,y2] is
        inside the named ROI.
        """
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return self.point_in_roi(cx, cy, roi_name)

    def get_active_rois_for_box(
        self,
        x1: int, y1: int, x2: int, y2: int,
    ) -> list[str]:
        """Return a list of ROI names that contain the box centre."""
        return [
            name for name in self.rois
            if self.box_centre_in_roi(x1, y1, x2, y2, name)
        ]

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def draw_rois(self, frame, alpha: float = 0.25):
        """
        Draw all ROIs on *frame* as semi-transparent filled polygons
        with labelled outlines.
        """
        overlay = frame.copy()
        colors = [
            (0, 255, 255),   # yellow
            (255, 0, 255),   # magenta
            (0, 255, 0),     # green
            (255, 128, 0),   # orange
        ]
        for i, (name, pts) in enumerate(self.rois.items()):
            color = colors[i % len(colors)]
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], True, color, 2)
            # Label at top-left vertex
            lx, ly = pts[0]
            cv2.putText(
                frame, name.upper().replace("_", " "),
                (int(lx), int(ly) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

