import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch

# Paths ====
VIDEO_PATH = r"D:\OneDrive\Desktop\Assignment_Suvit_Kumar\conversion_file_fixed.mp4"
PROJECT_ROOT = "yolo_sam_precise_pipeline"
FRAMES_DIR = os.path.join(PROJECT_ROOT, "frames")
ANNOTATED_DIR = os.path.join(PROJECT_ROOT, "annotated")
ANNOTATIONS_JSON = os.path.join(PROJECT_ROOT, "annotations.json")
ANNOTATED_VIDEO_PATH = os.path.join(PROJECT_ROOT, "annotated_output.mp4")

# Setup ====
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Extract Frames 
def extract_frames(video_path, output_dir, fps=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open video.")
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(original_fps / fps)
    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
        count += 1
    cap.release()
    print(f"Extracted {saved} frames.")

# Load Models 
yolo_model = YOLO("yolov8n-seg.pt")  # Lightweight version
sam = sam_model_registry["vit_b"](checkpoint=r"D:\OneDrive\Desktop\Assignment_Suvit_Kumar\sam_vit_b_01ec64.pth")
sam.to("cpu")
mask_generator = SamAutomaticMaskGenerator(sam)

TARGET_CLASSES = ['car', 'person', 'bicycle']

# Helper: IoU 
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# Annotate Frames 
def annotate_frames():
    annotations = []
    for fname in sorted(os.listdir(FRAMES_DIR)):
        frame_path = os.path.join(FRAMES_DIR, fname)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        result = yolo_model(frame, conf=0.4)[0]
        objects = []

        # Run SAM on full frame
        masks = mask_generator.generate(frame)

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            if class_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            best_iou = 0
            best_mask = None

            for m in masks:
                m_box = m["bbox"]  # [x, y, w, h]
                mx1, my1 = m_box[0], m_box[1]
                mx2, my2 = mx1 + m_box[2], my1 + m_box[3]
                iou = compute_iou([x1, y1, x2, y2], [mx1, my1, mx2, my2])
                if iou > best_iou:
                    best_iou = iou
                    best_mask = m["segmentation"]

            if best_mask is None or best_iou < 0.1:
                continue

            mask_np = best_mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                polygon = contour[:, 0, :].tolist()
                if len(polygon) >= 3:
                    objects.append({"class": class_name, "polygon": polygon})
                    cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], True, (0, 255, 0), 2)
                    cv2.putText(frame, class_name, polygon[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if objects:
            annotations.append({"frame": fname, "objects": objects})
            cv2.imwrite(os.path.join(ANNOTATED_DIR, fname), frame)
        else:
            print(f"🚫 No valid objects in {fname}")

    with open(ANNOTATIONS_JSON, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f" Saved {len(annotations)} annotated frames to JSON.")

# Create Video from Annotated Frames 
def create_video_from_frames(image_folder, output_path, fps=3):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    if not images:
        print("⚠️ No images to compile into video.")
        return

    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"🎬 Annotated video saved as: {output_path}")

# Run Pipeline 
extract_frames(VIDEO_PATH, FRAMES_DIR)
annotate_frames()
create_video_from_frames(ANNOTATED_DIR, ANNOTATED_VIDEO_PATH)
print(" All Done: YOLO + SAM with polygon + video output.")
