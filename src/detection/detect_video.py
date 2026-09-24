from ultralytics import YOLO

# Load YOLO11 Nano
model = YOLO("yolo11n.pt")

# Run object detection on KTM traffic video
model.predict(
    source="data/videos/ktm_trafic.mp4",
    save=True,
    conf=0.35
)

print("Detection completed successfully!")
