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
  - Output video rendering (OpenCV + FFmpeg)

Public API
----------
process_video(video_path, output_path, analysis_id, roi_config, progress_cb)
    → returns a result dict with counts, density, paths, etc.
"""

import os
import sys
import time
import uuid
import subprocess

import cv2
import numpy as np

# Make project root importable regardless of working directory
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)

import config
from src.utils.helpers import get_device, draw_label, get_class_color, ensure_dir
from src.analytics.counter import TrafficCounter
from src.analytics.density import (
    DensityTracker,
    calculate_density,
    density_color_bgr,
)
from src.analytics.statistics import save_timeseries_csv, save_analysis_to_db
from src.analytics.roi import ROIManager

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "Ultralytics not installed. Run: pip install ultralytics"
    ) from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_overlay(
    frame,
    frame_idx: int,
    fps: float,
    counts: dict,
    density: str,
    vehicle_in_frame: int,
):
    """Draw a semi-transparent stats panel on the top-left corner."""

    panel_h = 180
    panel_w = 330

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (5, 5),
        (panel_w, panel_h),
        (20, 20, 20),
        cv2.FILLED,
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
        frame,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    # lines = [
    #     ("AI Traffic Monitor", 0.50, (200, 200, 200)),
    #     (
    #         f"Frame: {frame_idx:05d}  FPS: {fps:.1f}",
    #         0.40,
    #         (180, 180, 180),
    #     ),
    #     (
    #         f"Vehicles (frame): {vehicle_in_frame}",
    #         0.40,
    #         (255, 255, 255),
    #     ),
    #     (
    #         f"Density: {density}",
    #         0.45,
    #         density_color_bgr(density),
    #     ),
    #     (
    #         f"Total Vehicles:  {counts.get('total_vehicles', 0)}",
    #         0.42,
    #         (255, 255, 255),
    #     ),
    #     (
    #         f"Cars: {counts.get('cars', 0)}  "
    #         f"Motos: {counts.get('motorcycles', 0)}",
    #         0.40,
    #         (200, 230, 255),
    #     ),
    #     (
    #         f"Buses: {counts.get('buses', 0)}  "
    #         f"Trucks: {counts.get('trucks', 0)}",
    #         0.40,
    #         (200, 230, 255),
    #     ),
    #     (
    #         f"Pedestrians: {counts.get('pedestrians', 0)}",
    #         0.40,
    #         (255, 220, 180),
    #     ),
    # ]
    lines = [
    ("AI TRAFFIC MONITOR", 0.50, (200, 200, 200)),
    (f"Frame: {frame_idx:05d}  FPS: {fps:.1f}", 0.40, (180, 180, 180)),
    (f"Vehicles in frame: {vehicle_in_frame}", 0.40, (255, 255, 255)),
    (f"Density: {density}", 0.45, density_color_bgr(density)),
    (f"Unique vehicles: {counts.get('total_vehicles', 0)}", 0.40, (255, 255, 255)),
    (f"Cars: {counts.get('cars', 0)}  Motorcycles: {counts.get('motorcycles', 0)}", 0.38, (200, 230, 255)),
    (f"Buses: {counts.get('buses', 0)}  Trucks: {counts.get('trucks', 0)}", 0.38, (200, 230, 255)),
    (f"Pedestrians: {counts.get('pedestrians', 0)}", 0.40, (255, 220, 180)),
]

    y = 24

    for text, scale, color in lines:
        cv2.putText(
            frame,
            text,
            (10, y),
            font,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )

        y += 20


def _draw_detection(
    frame,
    x1,
    y1,
    x2,
    y2,
    class_id: int,
    display_id,
    conf: float,
    class_name: str,
    active_rois: list[str],
):
    """Draw bounding box + label for one detection."""

    color = get_class_color(class_id)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    # Example:
    # CAR | ID:001
    # label = f"{class_name} | ID: {track_id}"
    label = f"{class_name} #{display_id:03d}"

    if active_rois:
        label += f" [{', '.join(active_rois)}]"

    draw_label(
        frame,
        label,
        x1,
        y1 - 2,
        color=color,
        bg_color=(20, 20, 20),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    output_video_path: str,
    csv_report_path: str,
    analysis_id: str,
    roi_config: dict | None = None,
    progress_callback=None,
    conf_threshold: float | None = None,
    user_id: int | None = None,
) -> dict:
    """
    Run the full detection → tracking → analytics → output pipeline.

    Parameters
    ----------
    video_path : str
        Path to the input MP4 video.

    output_video_path : str
        Final path of the annotated H.264 video.

    csv_report_path : str
        Where to write the per-frame time-series CSV.

    analysis_id : str
        Unique ID for this run.

    roi_config : dict, optional
        Named ROI polygons, e.g.
        {
            "zebra_crossing": [
                (x1, y1),
                (x2, y2),
                ...
            ]
        }

    progress_callback : callable, optional
        Called with (current_frame, total_frames).

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

    # -----------------------------------------------------------------------
    # Validate input
    # -----------------------------------------------------------------------

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # -----------------------------------------------------------------------
    # Ensure output directories exist
    # -----------------------------------------------------------------------

    ensure_dir(os.path.dirname(output_video_path))
    ensure_dir(os.path.dirname(csv_report_path))

    # -----------------------------------------------------------------------
    # Load YOLO model
    # -----------------------------------------------------------------------

    device = get_device()

    model = YOLO(config.MODEL_PATH)

    # Warm up MPS / YOLO.
    # Ultralytics expects one image with shape (H, W, C).
    _ = model(
        np.zeros(
            (640, 640, 3),
            dtype=np.uint8,
        ),
        verbose=False,
    )

    # -----------------------------------------------------------------------
    # Open input video
    # -----------------------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    total_frames = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        or 1
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration_sec = total_frames / src_fps

    # -----------------------------------------------------------------------
    # Set up temporary OpenCV output
    # -----------------------------------------------------------------------
    #
    # OpenCV writes an intermediate MP4 using mp4v.
    # FFmpeg will convert this file to H.264 after processing.
    #

    temp_video_path = (
        os.path.splitext(output_video_path)[0]
        + "_opencv.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        temp_video_path,
        fourcc,
        src_fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create temporary output video: "
            f"{temp_video_path}"
        )

    # -----------------------------------------------------------------------
    # Analytics objects
    # -----------------------------------------------------------------------

    counter = TrafficCounter()

    density_tracker = DensityTracker()

    roi_manager = (
        ROIManager(roi_config)
        if roi_config
        else None
    )
    # Clean sequential IDs for display only.
    # ByteTrack's original IDs are still kept internally.
    display_id_map = {}
    next_display_id = 1
    previous_centers = {}

    # -----------------------------------------------------------------------
    # Processing variables
    # -----------------------------------------------------------------------

    frame_idx = 0

    t_start = time.perf_counter()

    fps_display = 0.0

    # -----------------------------------------------------------------------
    # YOLO + ByteTrack
    # -----------------------------------------------------------------------

    results_gen = model.track(
        source=video_path,
        tracker=config.TRACKER_CONFIG,
        conf=conf_threshold,
        classes=config.CLASSES_OF_INTEREST,
        stream=True,
        verbose=False,
        device=device,
    )

    try:

        for result in results_gen:

            frame = result.orig_img.copy()

            frame_idx += 1
            # ---------------------------------------------------------------
            # Virtual counting line
            # ---------------------------------------------------------------

            frame_height, frame_width = frame.shape[:2]

            counting_line_y = int(
                frame_height * config.COUNTING_LINE_POSITION
            )

            cv2.line(
                frame,
                (0, counting_line_y),
                (frame_width, counting_line_y),
                (0, 255, 255),
                2,
            )

            # ---------------------------------------------------------------
            # Running FPS
            # ---------------------------------------------------------------

            if frame_idx % 10 == 0:

                elapsed = (
                    time.perf_counter()
                    - t_start
                )

                fps_display = (
                    frame_idx / elapsed
                    if elapsed > 0
                    else 0.0
                )

            # ---------------------------------------------------------------
            # Parse detections
            # ---------------------------------------------------------------

            vehicles_in_frame = 0

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                boxes = result.boxes

                for i in range(len(boxes)):

                    # -------------------------------------------------------
                    # Bounding box
                    # -------------------------------------------------------

                    xyxy = (
                        boxes.xyxy[i]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    x1, y1, x2, y2 = xyxy

                    # -------------------------------------------------------
                    # Class + confidence
                    # -------------------------------------------------------

                    class_id = int(
                        boxes.cls[i]
                        .cpu()
                        .item()
                    )

                    conf = float(
                        boxes.conf[i]
                        .cpu()
                        .item()
                    )

                    # -------------------------------------------------------
                    # Tracking ID
                    # -------------------------------------------------------

                    track_id = None

                    if boxes.id is not None:

                        track_id = int(
                            boxes.id[i]
                            .cpu()
                            .item()
                        )

                    # Skip detections without tracking IDs.
                    if track_id is None:
                        continue

                    # -------------------------------------------------------
                    # Class name
                    # -------------------------------------------------------

                    # Class name
                    # -------------------------------------------------------

                    # Use the most frequently observed class for this track.
                    # This reduces BUS/TRUCK switching caused by individual
                    # frame-level YOLO predictions.

                    stable_class_id = counter.get_stable_class(track_id)

                    if stable_class_id is None:
                        stable_class_id = class_id

                    class_name = config.CLASS_NAMES.get(
                        stable_class_id,
                        f"CLASS_{stable_class_id}",
                    )

                    # Clean sequential IDs for display only.
                    # ByteTrack's original IDs are still kept internally.
                    if track_id not in display_id_map:
                        display_id_map[track_id] = next_display_id
                        next_display_id += 1

                    display_id = display_id_map[track_id]
                    # -------------------------------------------------------
                    # Register unique detected object
                    # -------------------------------------------------------

                    counter.update(
                        track_id,
                        class_id,
                    )
                   # -------------------------------------------------------
                    # Virtual counting-line detection
                    # -------------------------------------------------------

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    previous_center = previous_centers.get(track_id)

                    if previous_center is not None:

                        previous_y = previous_center[1]

                        # Detect crossing in either direction.
                        crossed_line = (
                            (previous_y < counting_line_y <= center_y)
                            or
                            (previous_y > counting_line_y >= center_y)
                        )

                        if crossed_line:

                            counter.register_crossing(track_id)

                    # Save current position for the next frame.
                    previous_centers[track_id] = (
                        center_x,
                        center_y,
                    )

                    # -------------------------------------------------------
                    # Per-frame vehicle count
                    # -------------------------------------------------------

                    if (
                        class_id
                        in config.VEHICLE_CLASS_IDS
                    ):
                        vehicles_in_frame += 1

                    # -------------------------------------------------------
                    # ROI check
                    # -------------------------------------------------------

                    active_rois = []

                    if roi_manager:

                        active_rois = (
                            roi_manager
                            .get_active_rois_for_box(
                                x1,
                                y1,
                                x2,
                                y2,
                            )
                        )

                    # -------------------------------------------------------
                    # Draw detection
                    # -------------------------------------------------------

                    _draw_detection(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        class_id,
                        display_id,
                        conf,
                        class_name,
                        active_rois,
                    )

            # ---------------------------------------------------------------
            # Traffic density
            # ---------------------------------------------------------------

            density = calculate_density(
                vehicles_in_frame
            )

            # ---------------------------------------------------------------
            # Time-series sampling
            # ---------------------------------------------------------------

            if (
                frame_idx
                % config.STATS_SAMPLE_EVERY_N_FRAMES
                == 0
            ):

                density_tracker.record(
                    frame_idx,
                    vehicles_in_frame,
                )

            # ---------------------------------------------------------------
            # ROI overlay
            # ---------------------------------------------------------------

            if roi_manager:

                roi_manager.draw_rois(
                    frame
                )

            # ---------------------------------------------------------------
            # Statistics overlay
            # ---------------------------------------------------------------

            # Unique objects detected throughout the video.
            counts = counter.get_counts()

            # Vehicles that have crossed the virtual counting line.
            crossing_counts = counter.get_crossing_counts()

            _draw_overlay(
                frame,
                frame_idx,
                fps_display,
                counts,
                density,
                vehicles_in_frame,
            )

            # ---------------------------------------------------------------
            # Write annotated frame
            # ---------------------------------------------------------------

            writer.write(frame)

            # ---------------------------------------------------------------
            # Progress callback
            # ---------------------------------------------------------------

            if (
                progress_callback
                and frame_idx % 30 == 0
            ):

                progress_callback(
                    frame_idx,
                    total_frames,
                )

    finally:

        # Always release OpenCV resources.
        cap.release()
        writer.release()

    # -----------------------------------------------------------------------
    # Convert temporary MP4 → browser-friendly H.264 MP4
    # -----------------------------------------------------------------------

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        temp_video_path,

        # H.264 video
        "-c:v",
        "libx264",

        # Browser-friendly pixel format
        "-pix_fmt",
        "yuv420p",

        # Helps browser playback start quickly.
        "-movflags",
        "+faststart",

        output_video_path,
    ]

    try:

        subprocess.run(
            ffmpeg_command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    except FileNotFoundError as e:

        # FFmpeg executable was not found.
        raise RuntimeError(
            "FFmpeg was not found. "
            "Please make sure FFmpeg is installed "
            "and available in the roadsafety environment."
        ) from e

    except subprocess.CalledProcessError as e:

        ffmpeg_error = (
            e.stderr.decode(
                "utf-8",
                errors="replace",
            )
            if e.stderr
            else "Unknown FFmpeg error."
        )

        raise RuntimeError(
            "FFmpeg failed to convert the processed video:\n"
            + ffmpeg_error
        ) from e

    finally:

        # Remove temporary OpenCV video.
        if os.path.exists(temp_video_path):

            try:
                os.remove(temp_video_path)

            except OSError:
                pass

    # -----------------------------------------------------------------------
    # Processing time
    # -----------------------------------------------------------------------

    t_end = time.perf_counter()

    processing_time = (
        t_end - t_start
    )

    # -----------------------------------------------------------------------
    # Final counts and density
    # -----------------------------------------------------------------------

    final_counts = counter.get_counts()
    final_crossing_counts = counter.get_crossing_counts()

    overall_density = (
        density_tracker
        .get_overall_density()
    )

    timeseries = (
        density_tracker.to_list()
    )

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------

    save_timeseries_csv(
        timeseries,
        csv_report_path,
    )

    # -----------------------------------------------------------------------
    # Save analysis to database
    # -----------------------------------------------------------------------

    save_analysis_to_db(
        analysis_id=analysis_id,
        video_filename=os.path.basename(
            video_path
        ),
        user_id=user_id,
        counts=final_counts,
        density=overall_density,
        fps=src_fps,
        total_frames=frame_idx,
        duration_sec=duration_sec,
        processing_time=processing_time,
        crossing_counts=final_crossing_counts,
        output_video=output_video_path,
        csv_report=csv_report_path,
        status="completed",
    )

    # -----------------------------------------------------------------------
    # Return results
    # -----------------------------------------------------------------------

    return {
        "counts": final_counts,
        "crossing_counts": final_crossing_counts,
        "density": overall_density,
        "fps": src_fps,
        "total_frames": frame_idx,
        "duration_sec": duration_sec,
        "processing_time": processing_time,
        "output_video": output_video_path,
        "csv_report": csv_report_path,
        "timeseries": timeseries,
    }