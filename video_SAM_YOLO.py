import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

#Config 
VIDEO_PATH = r"D:\OneDrive\Desktop\Assignment_Suvit_Kumar\conversion_file_fixed.mp4"
CHECKPOINT_PATH = r"D:\OneDrive\Desktop\Assignment_Suvit_Kumar\sam_vit_b_01ec64.pth"

PROJECT_ROOT = "yolo_sam_polygon_video"
FRAMES_DIR = os.path.join(PROJECT_ROOT, "frames")
ANNOTATIONS_JSON = os.path.join(PROJECT_ROOT, "annotations.json")
ANNOTATED_VIDEO = os.path.join(PROJECT_ROOT, "annotated_output.avi")

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(PROJECT_ROOT, exist_ok=True)

# Load Models
model = YOLO("yolov8n-seg.pt")  # Lightest version for speed
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH)
sam.to("cpu")
mask_generator = SamAutomaticMaskGenerator(sam)

TARGET_CLASSES = ['car', 'person', 'bicycle']

#Extract Frames
def extract_frames(video_path, fps=3):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(original_fps / fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []

    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append((f"frame_{saved:04d}.jpg", frame.copy()))
            saved += 1
        count += 1
    cap.release()
    print(f"Extracted {saved} frames.")
    return frames, frame_width, frame_height

#Annotate with Polygons 
def annotate_and_generate_video(frames, width, height):
    annotations = []

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(ANNOTATED_VIDEO, fourcc, 3, (width, height))

    for fname, frame in frames:
        result = model(frame, conf=0.4)[0]
        objects = []

        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].int().cpu().numpy()
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            masks = mask_generator.generate(roi)
            if not masks:
                continue

            best_mask = max(masks, key=lambda m: m['area'])['segmentation']
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = best_mask.astype(np.uint8)

            contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                polygon = contour[:, 0, :].tolist()
                if len(polygon) >= 3:
                    objects.append({
                        "class": class_name,
                        "polygon": polygon
                    })
                    # Draw polygon
                    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                    cv2.putText(frame, class_name, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Save annotated frame to video
        out.write(frame)

        if objects:
            annotations.append({"frame": fname, "objects": objects})
        else:
            print(f"No valid objects in {fname}")

    out.release()
    print(f"🎥 Saved annotated video at: {ANNOTATED_VIDEO}")

    with open(ANNOTATIONS_JSON, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Saved {len(annotations)} frames to JSON")

# Run 
frames, w, h = extract_frames(VIDEO_PATH, fps=3)
annotate_and_generate_video(frames, w, h)
