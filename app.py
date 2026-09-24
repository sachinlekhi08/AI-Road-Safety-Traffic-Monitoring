"""
app.py
======
Flask backend for the AI Road Safety and Traffic Monitoring System.

Routes
------
GET  /                     → dashboard (index.html)
POST /upload               → upload a video, return {analysis_id, filename}
POST /analyze/<analysis_id>→ start background processing
GET  /status/<analysis_id> → poll processing status
GET  /results/<analysis_id>→ fetch final result dict (JSON)
GET  /history              → list all past analyses (JSON)
GET  /video/<analysis_id>  → stream the processed output video
GET  /videos               → list available built-in videos (JSON)
POST /analyze_existing     → analyse one of the built-in videos
GET  /timeseries/<id>      → fetch per-frame CSV as JSON array
"""

import os
import sys
import uuid
import threading
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, abort, send_file,
)
import config
from src.utils.helpers import safe_filename, allowed_video_extension, ensure_dir
from src.analytics.statistics import (
    get_analysis_by_id, get_all_analyses,
    save_analysis_to_db, init_database, update_analysis_status,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(ROOT, "dashboard", "templates"),
    static_folder=os.path.join(ROOT, "dashboard", "static"),
)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Ensure required directories exist
ensure_dir(config.UPLOAD_FOLDER)
ensure_dir(config.OUTPUT_VIDEO_FOLDER)
ensure_dir(config.OUTPUT_REPORT_FOLDER)
ensure_dir(os.path.dirname(config.DATABASE_PATH))
init_database()

# In-memory dict: analysis_id → {"status": str, "progress": int, "error": str}
_processing_state: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

def _run_processing(analysis_id: str, video_path: str, roi_config: dict | None):
    """Target function for the background processing thread."""
    _processing_state[analysis_id] = {"status": "processing", "progress": 0, "error": ""}

    def _progress(current, total):
        pct = int(100 * current / total) if total > 0 else 0
        _processing_state[analysis_id]["progress"] = pct

    try:
        from src.detection.processor import process_video

        out_video = os.path.join(config.OUTPUT_VIDEO_FOLDER, f"{analysis_id}_output.mp4")
        csv_path  = os.path.join(config.OUTPUT_REPORT_FOLDER, f"{analysis_id}_timeseries.csv")

        result = process_video(
            video_path=video_path,
            output_video_path=out_video,
            csv_report_path=csv_path,
            analysis_id=analysis_id,
            roi_config=roi_config,
            progress_callback=_progress,
        )
        _processing_state[analysis_id] = {
            "status": "completed",
            "progress": 100,
            "error": "",
        }
    except Exception:
        err = traceback.format_exc()
        _processing_state[analysis_id] = {
            "status": "failed",
            "progress": 0,
            "error": err,
        }
        update_analysis_status(analysis_id, "failed")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


# ---- Upload ---------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload_video():
    """
    Accept a video file upload.

    Returns JSON:
        {"analysis_id": "...", "filename": "...", "original_name": "..."}
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_video_extension(file.filename, config.ALLOWED_EXTENSIONS):
        return jsonify({"error": f"Unsupported format. Allowed: {config.ALLOWED_EXTENSIONS}"}), 400

    analysis_id = str(uuid.uuid4())
    saved_name  = f"{analysis_id}_{safe_filename(file.filename)}"
    save_path   = os.path.join(config.UPLOAD_FOLDER, saved_name)
    file.save(save_path)

    # Create a placeholder DB row so /status returns something immediately
    save_analysis_to_db(
        analysis_id=analysis_id,
        video_filename=file.filename,
        counts={},
        density="N/A",
        fps=0, total_frames=0, duration_sec=0, processing_time=0,
        status="uploaded",
    )
    _processing_state[analysis_id] = {"status": "uploaded", "progress": 0, "error": ""}

    return jsonify({
        "analysis_id": analysis_id,
        "filename": saved_name,
        "original_name": file.filename,
    })


# ---- Analyse uploaded video -----------------------------------------------

@app.route("/analyze/<analysis_id>", methods=["POST"])
def analyze(analysis_id: str):
    """
    Start background processing for a previously uploaded video.
    Body (JSON, optional):
        {"roi_config": {"zebra_crossing": [[x1,y1],[x2,y2],...]} }
    """
    state = _processing_state.get(analysis_id)
    if state is None:
        return jsonify({"error": "Unknown analysis_id"}), 404

    if state["status"] == "processing":
        return jsonify({"error": "Already processing"}), 409

    # Find the uploaded file
    uploads = os.listdir(config.UPLOAD_FOLDER)
    matching = [f for f in uploads if f.startswith(analysis_id)]
    if not matching:
        return jsonify({"error": "Uploaded video not found"}), 404

    video_path = os.path.join(config.UPLOAD_FOLDER, matching[0])

    body = request.get_json(silent=True) or {}
    roi_config = body.get("roi_config", None)

    t = threading.Thread(
        target=_run_processing,
        args=(analysis_id, video_path, roi_config),
        daemon=True,
    )
    t.start()

    return jsonify({"analysis_id": analysis_id, "status": "processing"})


# ---- Analyse a built-in video ---------------------------------------------

@app.route("/analyze_existing", methods=["POST"])
def analyze_existing():
    """
    Start analysis on one of the videos already in data/videos/.

    Body (JSON):
        {"filename": "ktm_trafic.mp4", "roi_config": ...}
    """
    body = request.get_json(silent=True) or {}
    filename = body.get("filename", "")
    if not filename:
        return jsonify({"error": "filename required"}), 400

    video_path = os.path.join(ROOT, "data", "videos", filename)
    if not os.path.isfile(video_path):
        return jsonify({"error": f"Video not found: {filename}"}), 404

    analysis_id = str(uuid.uuid4())
    roi_config  = body.get("roi_config", None)

    save_analysis_to_db(
        analysis_id=analysis_id,
        video_filename=filename,
        counts={},
        density="N/A",
        fps=0, total_frames=0, duration_sec=0, processing_time=0,
        status="queued",
    )
    _processing_state[analysis_id] = {"status": "queued", "progress": 0, "error": ""}

    t = threading.Thread(
        target=_run_processing,
        args=(analysis_id, video_path, roi_config),
        daemon=True,
    )
    t.start()

    return jsonify({"analysis_id": analysis_id, "status": "processing"})


# ---- Status ---------------------------------------------------------------

@app.route("/status/<analysis_id>")
def status(analysis_id: str):
    """Poll processing status. Returns JSON with status + progress."""
    state = _processing_state.get(analysis_id)
    if state is None:
        # Fall back to DB
        row = get_analysis_by_id(analysis_id)
        if row:
            return jsonify({"status": row["status"], "progress": 100, "error": ""})
        return jsonify({"error": "Unknown analysis_id"}), 404
    return jsonify(state)


# ---- Results --------------------------------------------------------------

@app.route("/results/<analysis_id>")
def results(analysis_id: str):
    """Return the full analysis result from the database."""
    row = get_analysis_by_id(analysis_id)
    if not row:
        return jsonify({"error": "No results found"}), 404
    return jsonify(row)


# ---- History --------------------------------------------------------------

@app.route("/history")
def history():
    """Return all past analyses (newest first)."""
    rows = get_all_analyses()
    return jsonify(rows)


# ---- Time-series data -----------------------------------------------------

@app.route("/timeseries/<analysis_id>")
def timeseries(analysis_id: str):
    """Return the per-frame CSV data as a JSON array."""
    import csv
    csv_path = os.path.join(config.OUTPUT_REPORT_FOLDER, f"{analysis_id}_timeseries.csv")
    if not os.path.isfile(csv_path):
        return jsonify([])
    data = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "frame": int(row["frame"]),
                "vehicle_count": int(row["vehicle_count"]),
                "density": row["density"],
            })
    return jsonify(data)


# ---- Processed video stream -----------------------------------------------

@app.route("/video/<analysis_id>")
def serve_video(analysis_id: str):
    """Stream the processed output video."""
    filename = f"{analysis_id}_output.mp4"
    video_path = os.path.join(config.OUTPUT_VIDEO_FOLDER, filename)
    if not os.path.isfile(video_path):
        abort(404)
    return send_file(video_path, mimetype="video/mp4", conditional=True)


# ---- Built-in video list --------------------------------------------------

@app.route("/videos")
def list_videos():
    """Return a list of videos available in data/videos/."""
    videos_dir = os.path.join(ROOT, "data", "videos")
    if not os.path.isdir(videos_dir):
        return jsonify([])
    videos = [
        f for f in os.listdir(videos_dir)
        if os.path.splitext(f)[1].lower().lstrip(".") in config.ALLOWED_EXTENSIONS
    ]
    return jsonify(sorted(videos))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True,   # required for background processing threads
    )

