"""
config.py
=========
Central configuration for the AI Road Safety and Traffic Monitoring System.
All key parameters are defined here so they can be changed in one place
without hunting through the codebase.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")

# Where uploaded videos land
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# Processed output videos
OUTPUT_VIDEO_FOLDER = os.path.join(BASE_DIR, "outputs", "videos")

# CSV / report output
OUTPUT_REPORT_FOLDER = os.path.join(BASE_DIR, "outputs", "reports")

# SQLite database
DATABASE_PATH = os.path.join(BASE_DIR, "database", "traffic.db")

# ---------------------------------------------------------------------------
# Detection / tracking
# ---------------------------------------------------------------------------

# Minimum confidence score to accept a detection (0–1)
CONFIDENCE_THRESHOLD = 0.25

# ByteTrack configuration shipped with Ultralytics
TRACKER_CONFIG = "bytetrack.yaml"

# COCO class IDs we care about
# 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 9=traffic light
CLASSES_OF_INTEREST = [0, 1, 2, 3, 5, 7, 9]

# Human-readable labels for the above class IDs
CLASS_NAMES = {
    0: "PERSON",
    1: "BICYCLE",
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK",
    9: "TRAFFIC LIGHT",
}

# Which class IDs count as "vehicles" for counting/density purposes
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]

# ---------------------------------------------------------------------------
# Traffic density thresholds  (vehicles visible in a single frame)
# ---------------------------------------------------------------------------
# These numbers are intentionally conservative for a typical traffic-camera
# field of view.  Adjust to suit the actual videos being analysed.
#
#   0 – LOW_THRESHOLD  vehicles  → LOW density
#   LOW_THRESHOLD – HIGH_THRESHOLD vehicles  → MEDIUM density
#   > HIGH_THRESHOLD vehicles  → HIGH density
DENSITY_LOW_THRESHOLD = 5     # ≤ 5 vehicles/frame  → LOW
DENSITY_HIGH_THRESHOLD = 15   # > 15 vehicles/frame → HIGH

# ---------------------------------------------------------------------------
# Video upload limits
# ---------------------------------------------------------------------------
MAX_CONTENT_LENGTH = 500 * 1024 * 1024   # 500 MB hard cap
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
# Resize frames before inference to save memory/time (width, height).
# Set to None to keep original resolution.
FRAME_RESIZE = None   # e.g. (1280, 720)

# How often (in frames) to save a stats snapshot for the time-series chart.
# Lower = more data points; higher = faster processing.
STATS_SAMPLE_EVERY_N_FRAMES = 10

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
SECRET_KEY = "road-safety-dev-key-change-in-production"
DEBUG = True
HOST = "0.0.0.0"
PORT = 5001

