"""
src/detection/processor.py
===========================
Core video-processing pipeline.

This module ties together:
  - YOLO11 detection (Ultralytics)
  - ByteTrack tracking
  - Traffic counting  (counter.py)
  - Traffic density   (density.py)
  - ROI checks        (roi.py)
  - Result storage    (statistics.py)
  - Output video rendering (OpenCV)

Public API
----------
process_video(video_path, output_path, analysis_id, roi_config, progress_cb)
    → returns a result dict with counts, density, paths, etc.
"""

import os
import sys
import time
import uuid
import cv2
import numpy as np

# Make project root importable regardless of working directory
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

import config
from src.utils.helpers import get_device, draw_label, get_class_color, ensure_dir
from src.analytics.counter import TrafficCounter
from src.analytics.density import DensityTracker, calculate_density, density_color_bgr
from src.analytics.statistics import save_timeseries_csv, save_analysis_to_db
from src.analytics.roi import ROIManager

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("Ultralytics not installed. Run: pip install ultralytics") from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_overlay(frame, frame_idx: int, fps: float,
                  counts: dict, density: str,
                  vehicle_in_frame: int):
    """Draw a semi-transparent stats panel on the top-left corner."""
    panel_h = 180
    panel_w = 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    dy = cv2.FONT_HERSHEY_SIMPLEX  # just using for clarity; dy is set below
    lines = [
        ("AI Traffic Monitor", 0.50, (200, 200, 200)),
        (f"Frame: {frame_idx:05d}  FPS: {fps:.1f}", 0.40, (180, 180, 180)),
        (f"Vehicles (frame): {vehicle_in_frame}", 0.42, (255, 255, 255)),
        (f"Density: {density}", 0.45, density_color_bgr(density)),
        (f"Total Vehicles:  {counts.get('total_vehicles', 0)}", 0.42, (255, 255, 255)),
        (f"Cars: {counts.get('cars',0)}  Motos: {counts.get('motorcycles',0)}", 0.40, (200, 230, 255)),
        (f"Buses: {counts.get('buses',0)}  Trucks: {counts.get('trucks',0)}", 0.40, (200, 230, 255)),
        (f"Pedestrians: {counts.get('pedestrians',0)}", 0.40, (255, 220, 180)),
    ]
    y = 24
    for text, scale, color in lines:
        cv2.putText(frame, text, (10, y), font, scale, color, 1, cv2.LINE_AA)
        y += 20


def _draw_detection(frame, x1, y1, x2, y2,
                    class_id: int, track_id: int,
                    conf: float, class_name: str,
                    active_rois: list[str]):
    """Draw bounding box + label for one detection."""
    color = get_class_color(class_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label:  CAR | ID: 12
    label = f"{class_name} | ID: {track_id}"
    if active_rois:
        label += f" [{', '.join(active_rois)}]"

    draw_label(frame, label, x1, y1 - 2, color=color, bg_color=(20, 20, 20))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    output_video_path: str,
    csv_report_path: str,
    analysis_id: str,
    roi_config: dict | None = None,
    progress_callback=None,        # callable(current_frame, total_frames)
    conf_threshold: float | None = None,
) -> dict:
    """
    Run the full detection → tracking → analytics → output pipeline.

    Parameters
    ----------
    video_path : str
        Path to the input MP4 video.
    output_video_path : str
        Where to write the annotated output video.
    csv_report_path : str
        Where to write the per-frame time-series CSV.
    analysis_id : str
        Unique ID for this run (used for DB record).
    roi_config : dict, optional
        Named ROI polygons, e.g.
        {"zebra_crossing": [(x1,y1), (x2,y2), …]}.
    progress_callback : callable, optional
        Called with (current_frame, total_frames) for progress reporting.
    conf_threshold : float, optional
        Override config.CONFIDENCE_THRESHOLD.

    Returns
    -------
    dict
        {
            "counts": {...},
            "density": "LOW/MEDIUM/HIGH",
            "fps": float,
            "total_frames": int,
            "duration_sec": float,
            "processing_time": float,
            "output_video": str,
            "csv_report": str,
            "timeseries": [...],
        }
    """
    if conf_threshold is None:
        conf_threshold = config.CONFIDENCE_THRESHOLD

    # -- Validate input --
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # -- Ensure output dirs exist --
    ensure_dir(os.path.dirname(output_video_path))
    ensure_dir(os.path.dirname(csv_report_path))

    # -- Load model --
    device = get_device()
    model = YOLO(config.MODEL_PATH)
    # Warm up (optional, helps MPS initialise)
    _ = model(np.zeros((1, 640, 640, 3), dtype=np.uint8), verbose=False)

    # -- Open video --
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / src_fps

    # -- Set up output video writer --
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, src_fps, (width, height))

    # -- Analytics objects --
    counter = TrafficCounter()
    density_tracker = DensityTracker()
    roi_manager = ROIManager(roi_config) if roi_config else None

    frame_idx = 0
    t_start = time.perf_counter()
    fps_display = 0.0

    # Use YOLO's track() generator for memory efficiency
    results_gen = model.track(
        source=video_path,
        tracker=config.TRACKER_CONFIG,
        conf=conf_threshold,
        classes=config.CLASSES_OF_INTEREST,
        stream=True,          # process frame-by-frame (memory efficient)
        verbose=False,
        device=device,
    )

    try:
        for result in results_gen:
            frame = result.orig_img.copy()
            frame_idx += 1

            # Running FPS (updated every 10 frames to avoid flicker)
            if frame_idx % 10 == 0:
                elapsed = time.perf_counter() - t_start
                fps_display = frame_idx / elapsed if elapsed > 0 else 0.0

            # ---- Parse detections ----
            vehicles_in_frame = 0
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    # Coordinates (xyxy) as integers
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    class_id = int(boxes.cls[i].cpu().item())
                    conf     = float(boxes.conf[i].cpu().item())

                    # Tracking ID (None if tracker lost it)
                    track_id = None
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].cpu().item())

                    if track_id is None:
                        continue  # skip untracked boxes

                    class_name = config.CLASS_NAMES.get(class_id, f"CLASS_{class_id}")

                    # Count unique objects
                    counter.update(track_id, class_id)

                    # Per-frame vehicle count (for density)
                    if class_id in config.VEHICLE_CLASS_IDS:
                        vehicles_in_frame += 1

                    # ROI check
                    active_rois = []
                    if roi_manager:
                        active_rois = roi_manager.get_active_rois_for_box(x1, y1, x2, y2)

                    # Draw detection on frame
                    _draw_detection(frame, x1, y1, x2, y2,
                                    class_id, track_id, conf,
                                    class_name, active_rois)

            # ---- Density ----
            density = calculate_density(vehicles_in_frame)

            # Sample time-series every N frames
            if frame_idx % config.STATS_SAMPLE_EVERY_N_FRAMES == 0:
                density_tracker.record(frame_idx, vehicles_in_frame)

            # ---- ROI overlay ----
            if roi_manager:
                roi_manager.draw_rois(frame)

            # ---- Stats panel overlay ----
            counts = counter.get_counts()
            _draw_overlay(frame, frame_idx, fps_display, counts, density, vehicles_in_frame)

            # Write annotated frame
            writer.write(frame)

            # Progress callback
            if progress_callback and frame_idx % 30 == 0:
                progress_callback(frame_idx, total_frames)

    finally:
        cap.release()
        writer.release()

    t_end = time.perf_counter()
    processing_time = t_end - t_start

    # ---- Final counts and density ----
    final_counts = counter.get_counts()
    overall_density = density_tracker.get_overall_density()
    timeseries = density_tracker.to_list()

    # ---- Save CSV ----
    save_timeseries_csv(timeseries, csv_report_path)

    # ---- Save to DB ----
    save_analysis_to_db(
        analysis_id=analysis_id,
        video_filename=os.path.basename(video_path),
        counts=final_counts,
        density=overall_density,
        fps=src_fps,
        total_frames=frame_idx,
        duration_sec=duration_sec,
        processing_time=processing_time,
        output_video=output_video_path,
        csv_report=csv_report_path,
        status="completed",
    )

    return {
        "counts": final_counts,
        "density": overall_density,
        "fps": src_fps,
        "total_frames": frame_idx,
        "duration_sec": duration_sec,
        "processing_time": processing_time,
        "output_video": output_video_path,
        "csv_report": csv_report_path,
        "timeseries": timeseries,
    }

