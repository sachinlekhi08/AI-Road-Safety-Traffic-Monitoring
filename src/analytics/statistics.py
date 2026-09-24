"""
src/analytics/statistics.py
============================
Collects per-frame statistics and saves them to CSV and SQLite.

Statistics collected
--------------------
- Per-frame vehicle counts (sampled every N frames)
- Density labels over time
- Final aggregate counts

Saved outputs
-------------
- <output_dir>/reports/<analysis_id>_timeseries.csv   (time-series data)
- database/traffic.db  (SQLite: one row per analysis run)
"""

import csv
import sqlite3
import os
import datetime
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_timeseries_csv(timeseries: list[dict], output_path: str) -> str:
    """
    Write the per-frame stats to a CSV file.

    Parameters
    ----------
    timeseries : list of dict
        Each dict has keys: frame, vehicle_count, density.
    output_path : str
        Full path for the CSV file.

    Returns
    -------
    str : the output_path on success.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["frame", "vehicle_count", "density"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timeseries)
    return output_path


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def init_database(db_path: str = config.DATABASE_PATH):
    """
    Create the database and table if they do not already exist.
    Safe to call multiple times (idempotent).
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id     TEXT UNIQUE NOT NULL,
            video_filename  TEXT,
            created_at      TEXT,
            total_vehicles  INTEGER,
            cars            INTEGER,
            motorcycles     INTEGER,
            buses           INTEGER,
            trucks          INTEGER,
            bicycles        INTEGER,
            pedestrians     INTEGER,
            traffic_lights  INTEGER,
            traffic_density TEXT,
            fps             REAL,
            total_frames    INTEGER,
            duration_sec    REAL,
            processing_time REAL,
            status          TEXT,
            output_video    TEXT,
            csv_report      TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_analysis_to_db(
    analysis_id: str,
    video_filename: str,
    counts: dict,
    density: str,
    fps: float,
    total_frames: int,
    duration_sec: float,
    processing_time: float,
    output_video: str = "",
    csv_report: str = "",
    status: str = "completed",
    db_path: str = config.DATABASE_PATH,
) -> int:
    """
    Insert or update one analysis record in the SQLite database.

    Returns the row id.
    """
    init_database(db_path)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO analyses
            (analysis_id, video_filename, created_at,
             total_vehicles, cars, motorcycles, buses, trucks,
             bicycles, pedestrians, traffic_lights,
             traffic_density, fps, total_frames, duration_sec,
             processing_time, status, output_video, csv_report)
        VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        analysis_id,
        video_filename,
        now,
        counts.get("total_vehicles", 0),
        counts.get("cars", 0),
        counts.get("motorcycles", 0),
        counts.get("buses", 0),
        counts.get("trucks", 0),
        counts.get("bicycles", 0),
        counts.get("pedestrians", 0),
        counts.get("traffic_lights", 0),
        density,
        fps,
        total_frames,
        duration_sec,
        processing_time,
        status,
        output_video,
        csv_report,
    ))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_analysis_by_id(analysis_id: str, db_path: str = config.DATABASE_PATH) -> dict | None:
    """Fetch one analysis row as a dict, or None if not found."""
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_analyses(db_path: str = config.DATABASE_PATH) -> list[dict]:
    """Return all analysis records ordered by most recent first."""
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM analyses ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_analysis_status(
    analysis_id: str,
    status: str,
    db_path: str = config.DATABASE_PATH,
):
    """Update just the status field for a running/failed analysis."""
    init_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE analyses SET status=? WHERE analysis_id=?",
        (status, analysis_id),
    )
    conn.commit()
    conn.close()

