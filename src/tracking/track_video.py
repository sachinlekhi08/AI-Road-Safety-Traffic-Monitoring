from ultralytics import YOLO

# Load YOLO11 Nano
model = YOLO("yolo11n.pt")

# Run YOLO detection + ByteTrack tracking
model.track(
    source="data/videos/ktm_trafic.mp4",
    tracker="bytetrack.yaml",
    save=True,
    conf=0.35,
    classes=[0, 2, 3, 5, 7],
    verbose=True
)

print("Tracking completed successfully!")