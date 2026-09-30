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
            user_id             INTEGER,
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
    # Users table
    # -----------------------------------------------------------------------
    # Stores application users and their roles.
    # Passwords are stored as secure hashes, never as plain text.
    # -----------------------------------------------------------------------

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            username        TEXT UNIQUE NOT NULL,
            email           TEXT UNIQUE NOT NULL,
            password_hash   TEXT NOT NULL,
            role            TEXT NOT NULL DEFAULT 'user',
            created_at      TEXT NOT NULL
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
        "user_id": "INTEGER",
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
    user_id: int | None = None,
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
                user_id,
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
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        analysis_id,
        video_filename,
        user_id,
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
    user_id: int | None = None,
) -> list[dict]:
    """
    Return analysis records ordered by most recent first.

    Parameters
    ----------
    user_id : int | None
        If given, only that user's own analyses are returned.
        If None, every analysis is returned (used by the admin dashboard).
    """

    init_database(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()

    if user_id is not None:
        cur.execute(
            "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
    else:
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

    # ---------------------------------------------------------------------------
# User / authentication helpers
# ---------------------------------------------------------------------------

def create_user(
    username: str,
    email: str,
    password_hash: str,
    role: str = "user",
    db_path: str = config.DATABASE_PATH,
) -> int:
    """
    Create a new application user.

    Passwords must already be securely hashed before being passed here.

    Returns
    -------
    int
        SQLite row ID of the new user.
    """

    init_database(db_path)

    now = datetime.datetime.now().isoformat(timespec="seconds")

    conn = sqlite3.connect(db_path)

    cur = conn.cursor()

    cur.execute("""
        INSERT INTO users
            (username, email, password_hash, role, created_at)
        VALUES
            (?, ?, ?, ?, ?)
    """, (
        username,
        email,
        password_hash,
        role,
        now,
    ))

    conn.commit()

    user_id = cur.lastrowid

    conn.close()

    return user_id


def get_user_by_username(
    username: str,
    db_path: str = config.DATABASE_PATH,
) -> dict | None:
    """Return one user by username, or None if not found."""

    init_database(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,),
    )

    row = cur.fetchone()

    conn.close()

    return dict(row) if row else None

def get_user_by_email(email: str, db_path=config.DATABASE_PATH):
    """Return a user record by email address."""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        (email,),
    ).fetchone()

    conn.close()

    return dict(row) if row else None

def get_all_users(
    db_path: str = config.DATABASE_PATH,
) -> list[dict]:
    """Return all registered users."""

    init_database(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()

    cur.execute(
        "SELECT id, username, email, role, created_at "
        "FROM users ORDER BY created_at DESC"
    )

    rows = cur.fetchall()

    conn.close()

    return [dict(row) for row in rows]

#admin dashboard
def get_user_by_id(
    user_id: int,
    db_path: str = config.DATABASE_PATH,
) -> dict | None:
    """Return one user by numeric id, or None if not found."""

    init_database(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()

    conn.close()

    return dict(row) if row else None


def count_admins(db_path: str = config.DATABASE_PATH) -> int:
    """Return how many users currently have role='admin'."""

    init_database(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    count = cur.fetchone()[0]

    conn.close()

    return count


def update_user_role(
    user_id: int,
    role: str,
    db_path: str = config.DATABASE_PATH,
):
    """Change a user's role ('user' or 'admin')."""

    init_database(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    conn.commit()

    conn.close()


def delete_user(
    user_id: int,
    db_path: str = config.DATABASE_PATH,
):
    """Permanently delete a user account. Does not delete their analyses."""

    init_database(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()

    conn.close()


def delete_analysis(
    analysis_id: str,
    db_path: str = config.DATABASE_PATH,
):
    """Permanently delete one analysis record."""

    init_database(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,))
    conn.commit()

    conn.close()