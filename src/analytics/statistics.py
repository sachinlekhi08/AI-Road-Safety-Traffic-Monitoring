"""
src/analytics/statistics.py
============================
Collects per-frame statistics and saves them to CSV and SQLite.

Statistics collected
--------------------
- Per-frame vehicle counts (sampled every N frames)
- Density labels over time
- Final aggregate counts
- Traffic-flow / counting-line statistics

Saved outputs
-------------
- <output_dir>/reports/<analysis_id>_timeseries.csv
- database/traffic.db
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
    str
        The output_path on success.
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
    Create the database/table if needed.

    Also adds new traffic-flow columns to an existing database
    without deleting previous analysis records.
    """

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create the table for a new database.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id         TEXT UNIQUE NOT NULL,
            video_filename      TEXT,
            created_at          TEXT,
            total_vehicles      INTEGER,
            cars                INTEGER,
            motorcycles         INTEGER,
            buses               INTEGER,
            trucks              INTEGER,
            bicycles            INTEGER,
            pedestrians         INTEGER,
            traffic_lights      INTEGER,
            traffic_density     TEXT,
            fps                 REAL,
            total_frames        INTEGER,
            duration_sec        REAL,
            processing_time     REAL,

            crossed_vehicles    INTEGER DEFAULT 0,
            crossed_cars        INTEGER DEFAULT 0,
            crossed_motorcycles INTEGER DEFAULT 0,
            crossed_buses       INTEGER DEFAULT 0,
            crossed_trucks      INTEGER DEFAULT 0,

            status              TEXT,
            output_video        TEXT,
            csv_report          TEXT
        )
    """)

    # -----------------------------------------------------------------------
    # Database migration
    # -----------------------------------------------------------------------
    # If traffic.db already existed before the traffic-flow feature was
    # added, the new columns do not exist yet.
    #
    # Add them safely without deleting old analysis records.
    # -----------------------------------------------------------------------

    cur.execute("PRAGMA table_info(analyses)")
    existing_columns = {row[1] for row in cur.fetchall()}

    new_columns = {
        "crossed_vehicles": "INTEGER DEFAULT 0",
        "crossed_cars": "INTEGER DEFAULT 0",
        "crossed_motorcycles": "INTEGER DEFAULT 0",
        "crossed_buses": "INTEGER DEFAULT 0",
        "crossed_trucks": "INTEGER DEFAULT 0",
    }

    for column_name, column_type in new_columns.items():

        if column_name not in existing_columns:
            cur.execute(
                f"ALTER TABLE analyses ADD COLUMN "
                f"{column_name} {column_type}"
            )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Save analysis
# ---------------------------------------------------------------------------

def save_analysis_to_db(
    analysis_id: str,
    video_filename: str,
    counts: dict,
    density: str,
    fps: float,
    total_frames: int,
    duration_sec: float,
    processing_time: float,
    crossing_counts: dict | None = None,
    output_video: str = "",
    csv_report: str = "",
    status: str = "completed",
    db_path: str = config.DATABASE_PATH,
) -> int:
    """
    Insert or update one analysis record in the SQLite database.

    Parameters
    ----------
    crossing_counts : dict | None
        Traffic-flow statistics generated by TrafficCounter.

    Returns
    -------
    int
        SQLite row id.
    """

    if crossing_counts is None:
        crossing_counts = {}

    init_database(db_path)

    now = datetime.datetime.now().isoformat(timespec="seconds")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO analyses
            (
                analysis_id,
                video_filename,
                created_at,

                total_vehicles,
                cars,
                motorcycles,
                buses,
                trucks,
                bicycles,
                pedestrians,
                traffic_lights,

                traffic_density,
                fps,
                total_frames,
                duration_sec,
                processing_time,

                crossed_vehicles,
                crossed_cars,
                crossed_motorcycles,
                crossed_buses,
                crossed_trucks,

                status,
                output_video,
                csv_report
            )
        VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        analysis_id,
        video_filename,
        now,

        # ---------------------------------------------------------------
        # Unique detection statistics
        # ---------------------------------------------------------------
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

        # ---------------------------------------------------------------
        # Traffic-flow / counting-line statistics
        # ---------------------------------------------------------------
        crossing_counts.get("crossed_vehicles", 0),
        crossing_counts.get("crossed_cars", 0),
        crossing_counts.get("crossed_motorcycles", 0),
        crossing_counts.get("crossed_buses", 0),
        crossing_counts.get("crossed_trucks", 0),

        status,
        output_video,
        csv_report,
    ))

    conn.commit()

    row_id = cur.lastrowid

    conn.close()

    return row_id


# ---------------------------------------------------------------------------
# Get one analysis
# ---------------------------------------------------------------------------

def get_analysis_by_id(
    analysis_id: str,
    db_path: str = config.DATABASE_PATH,
) -> dict | None:
    """Fetch one analysis row as a dict, or None if not found."""

    init_database(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM analyses WHERE analysis_id = ?",
        (analysis_id,),
    )

    row = cur.fetchone()

    conn.close()

    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Get all analyses
# ---------------------------------------------------------------------------

def get_all_analyses(
    db_path: str = config.DATABASE_PATH,
) -> list[dict]:
    """Return all analysis records ordered by most recent first."""

    init_database(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM analyses ORDER BY created_at DESC"
    )

    rows = cur.fetchall()

    conn.close()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Update analysis status
# ---------------------------------------------------------------------------

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