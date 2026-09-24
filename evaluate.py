"""
evaluate.py
===========
Model evaluation script.

Runs YOLO11n on each traffic video in data/videos/ and reports:
  - Per-video detection examples
  - Confidence score statistics
  - Processing speed (FPS)
  - Total detections per class

IMPORTANT: This is NOT a formal mAP evaluation — that requires ground-truth
annotations which are not available.  Results are empirical observations for
reporting in a BSc project.

Usage
-----
    conda activate roadsafety
    python evaluate.py
    python evaluate.py --video data/videos/ktm_trafic.mp4
    python evaluate.py --conf 0.40 --max_frames 300
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import cv2
from ultralytics import YOLO
import config
from src.utils.helpers import get_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gather_videos(video_arg: str | None) -> list[Path]:
    """Return list of video paths to evaluate."""
    if video_arg:
        p = Path(video_arg)
        if not p.is_file():
            raise FileNotFoundError(f"Video not found: {video_arg}")
        return [p]
    videos_dir = Path(ROOT) / "data" / "videos"
    return sorted(videos_dir.glob("*.mp4"))


def run_evaluation(
    video_path: Path,
    model: YOLO,
    conf: float,
    max_frames: int | None,
    device: str,
) -> dict:
    """
    Run detection on *video_path* and return evaluation metrics.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frames_to_process = min(max_frames, tot_frames) if max_frames else tot_frames

    # Use Ultralytics stream generator
    results_gen = model.predict(
        source=str(video_path),
        conf=conf,
        classes=config.CLASSES_OF_INTEREST,
        stream=True,
        verbose=False,
        device=device,
    )

    class_detections = defaultdict(int)   # class_id → count
    conf_scores      = defaultdict(list)  # class_id → [conf, …]
    frame_counts     = []                 # detections per frame
    frame_idx        = 0
    t_start          = time.perf_counter()

    for result in results_gen:
        frame_idx += 1
        n_dets = 0
        if result.boxes is not None:
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].cpu().item())
                conf_val = float(result.boxes.conf[i].cpu().item())
                class_detections[cls_id] += 1
                conf_scores[cls_id].append(conf_val)
                n_dets += 1
        frame_counts.append(n_dets)
        if max_frames and frame_idx >= max_frames:
            break

    t_end    = time.perf_counter()
    elapsed  = t_end - t_start
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0

    # Build per-class stats
    class_stats = {}
    for cls_id in class_detections:
        scores = conf_scores[cls_id]
        class_stats[config.CLASS_NAMES.get(cls_id, f"class_{cls_id}")] = {
            "total_detections": class_detections[cls_id],
            "avg_confidence": round(sum(scores) / len(scores), 3),
            "min_confidence": round(min(scores), 3),
            "max_confidence": round(max(scores), 3),
        }

    total_dets = sum(class_detections.values())
    avg_per_frame = total_dets / frame_idx if frame_idx > 0 else 0

    return {
        "video": video_path.name,
        "resolution": f"{width}x{height}",
        "source_fps": round(src_fps, 1),
        "frames_processed": frame_idx,
        "total_frames": tot_frames,
        "processing_fps": round(proc_fps, 2),
        "elapsed_sec": round(elapsed, 2),
        "total_detections": total_dets,
        "avg_detections_per_frame": round(avg_per_frame, 2),
        "class_stats": class_stats,
    }


def print_report(metrics: dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print(f"  Video: {metrics['video']}")
    print("=" * 60)
    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return
    print(f"  Resolution    : {metrics['resolution']}")
    print(f"  Source FPS    : {metrics['source_fps']}")
    print(f"  Frames proc.  : {metrics['frames_processed']} / {metrics['total_frames']}")
    print(f"  Processing FPS: {metrics['processing_fps']}")
    print(f"  Elapsed time  : {metrics['elapsed_sec']} s")
    print(f"  Total detects : {metrics['total_detections']}")
    print(f"  Avg / frame   : {metrics['avg_detections_per_frame']}")
    print()
    print("  Per-class breakdown:")
    for cls_name, stats in metrics["class_stats"].items():
        print(f"    {cls_name:<16} | dets={stats['total_detections']:5d} "
              f"| avg_conf={stats['avg_confidence']:.3f} "
              f"| min={stats['min_confidence']:.3f} "
              f"| max={stats['max_confidence']:.3f}")


def save_csv_report(all_metrics: list[dict], out_path: str):
    """Save a flat CSV summary of all videos evaluated."""
    rows = []
    for m in all_metrics:
        if "error" in m:
            continue
        base = {
            "video": m["video"],
            "resolution": m["resolution"],
            "source_fps": m["source_fps"],
            "frames_processed": m["frames_processed"],
            "processing_fps": m["processing_fps"],
            "elapsed_sec": m["elapsed_sec"],
            "total_detections": m["total_detections"],
            "avg_detections_per_frame": m["avg_detections_per_frame"],
        }
        for cls_name, stats in m["class_stats"].items():
            base[f"{cls_name}_dets"]     = stats["total_detections"]
            base[f"{cls_name}_avg_conf"] = stats["avg_confidence"]
        rows.append(base)

    if not rows:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[+] CSV report saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11n on traffic videos."
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a single video (default: all in data/videos/)")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit frames per video (useful for quick tests)")
    args = parser.parse_args()

    device = get_device()
    print(f"[*] Using device: {device}")
    print(f"[*] Confidence threshold: {args.conf}")

    model  = YOLO(config.MODEL_PATH)
    videos = gather_videos(args.video)

    if not videos:
        print("[!] No videos found. Add MP4 files to data/videos/")
        return

    print(f"[*] Evaluating {len(videos)} video(s)…\n")

    all_metrics = []
    for vp in videos:
        print(f"[→] Processing: {vp.name}")
        metrics = run_evaluation(vp, model, args.conf, args.max_frames, device)
        print_report(metrics)
        all_metrics.append(metrics)

    # Save CSV
    out_csv = os.path.join(ROOT, "outputs", "reports", "evaluation_results.csv")
    save_csv_report(all_metrics, out_csv)

    print("\n[*] Evaluation complete.")
    print("[!] NOTE: These are empirical detection counts — not ground-truth mAP scores.")
    print("    Manual review of video frames is required to identify false positives/negatives.")


if __name__ == "__main__":
    main()

