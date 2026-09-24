"""
src/violations/violation_base.py
=================================
Base class for all road-safety violation modules.

Each violation detector should subclass ViolationDetectorBase and implement
the `check()` method.  The Flask backend and the processor call this uniform
interface so new detectors can be added without touching existing code.

Status of violation modules
----------------------------
- WrongWayDetector:     Placeholder – requires directional ROI setup.
- RedLightViolation:    Placeholder – requires traffic-light state module.
- HelmetDetector:       Placeholder – requires a custom-trained model.

None of these generate fake detections.  They are structured stubs ready
for future implementation.
"""


class ViolationDetectorBase:
    """Abstract base class for violation detectors."""

    name: str = "base"

    def check(self, track_id: int, class_id: int,
              x1: int, y1: int, x2: int, y2: int,
              frame_index: int, **kwargs) -> dict | None:
        """
        Examine one tracked detection and return a violation dict or None.

        Returns
        -------
        dict with keys {track_id, class_id, violation_type, frame, details}
        or None if no violation detected.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Stub implementations
# ---------------------------------------------------------------------------

class WrongWayDetector(ViolationDetectorBase):
    """
    PLACEHOLDER – Wrong-way detection.

    Full implementation requires:
    - A defined entry/exit ROI for the road.
    - Tracking the direction of movement of each vehicle.
    This can be added once the ROI system is configured for a specific video.
    """
    name = "wrong_way"

    def check(self, track_id, class_id, x1, y1, x2, y2, frame_index, **kwargs):
        # Not yet implemented – no fake detections generated.
        return None


class RedLightViolationDetector(ViolationDetectorBase):
    """
    PLACEHOLDER – Red-light violation detection.

    Full implementation requires:
    - Traffic-light state recognition (red/green/yellow).
    - A stop-line ROI.
    - Checking whether a vehicle crosses the stop line while light is red.
    """
    name = "red_light"

    def check(self, track_id, class_id, x1, y1, x2, y2, frame_index, **kwargs):
        return None


class HelmetDetector(ViolationDetectorBase):
    """
    PLACEHOLDER – Helmet / no-helmet detection.

    The pretrained COCO YOLO11n model does NOT reliably detect helmets.
    A custom fine-tuned model is required.  Architecture is prepared here
    so it can be swapped in when the model is available.
    """
    name = "helmet"

    def check(self, track_id, class_id, x1, y1, x2, y2, frame_index, **kwargs):
        return None


# ---------------------------------------------------------------------------
# Registry – add new detectors here
# ---------------------------------------------------------------------------

AVAILABLE_DETECTORS: list[ViolationDetectorBase] = [
    WrongWayDetector(),
    RedLightViolationDetector(),
    HelmetDetector(),
]


def run_all_detectors(
    track_id: int, class_id: int,
    x1: int, y1: int, x2: int, y2: int,
    frame_index: int,
    **kwargs,
) -> list[dict]:
    """
    Run every registered detector and return a list of violation dicts.
    Returns an empty list if no violations are detected.
    """
    violations = []
    for detector in AVAILABLE_DETECTORS:
        result = detector.check(
            track_id, class_id, x1, y1, x2, y2, frame_index, **kwargs
        )
        if result is not None:
            violations.append(result)
    return violations

