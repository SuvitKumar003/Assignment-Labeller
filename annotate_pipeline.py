# Yolo pipeline for Polygon Annotation used for detecting of cars in a video.
import os
import cv2
import numpy as np
import json
from ultralytics import YOLO

#Set up all paths


VIDEO_PATH = r"D:\OneDrive\Desktop\Assignment_Suvit_Kumar\conversion_file_fixed.mp4"

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f" Video not found: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(" OpenCV failed to open the video. Possible codec/format issue.")
else:
    print(" OpenCV successfully opened the video.")

print("🎞 Total frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("⏱ FPS:", cap.get(cv2.CAP_PROP_FPS))
cap.release()

#  Paths
PROJECT_ROOT = "polygon_annotation_pipeline"
FRAMES_DIR = os.path.join(PROJECT_ROOT, "frames")
MASKS_DIR = os.path.join(PROJECT_ROOT, "masks_png")
ANNOTATED_DIR = os.path.join(PROJECT_ROOT, "polygon_drawn")
ANNOTATIONS_JSON = os.path.join(PROJECT_ROOT, "annotations.json")

#  Clean folders for fresh start
for folder in [FRAMES_DIR, MASKS_DIR, ANNOTATED_DIR]:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    else:
        os.makedirs(folder)

print(" Paths Set:")
print(f" Input Video         : {VIDEO_PATH}")
print(f" Frames Folder       : {FRAMES_DIR}")
print(f" Mask PNGs           : {MASKS_DIR}")
print(f" Annotated Frames    : {ANNOTATED_DIR}")
print(f" Annotations JSON    : {ANNOTATIONS_JSON}")


#  STEP 2: Extract Frames


def extract_frames(video_path, output_dir, fps=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Could not open video. Check the path or file format.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f" Original FPS: {original_fps}, Total frames: {total_frames}")

    if fps is None or fps >= original_fps:
        frame_interval = 1
    else:
        frame_interval = int(original_fps / fps)

    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    cap.release()
    print(f" Extracted {saved} frames.")

extract_frames(VIDEO_PATH, FRAMES_DIR, fps=3)


# 🧠 STEP 3: Load YOLOv8 Model


model = YOLO("yolov8l-seg.pt")  
TARGET_CLASSES = ['car']  
annotations = []


#  STEP 4: Run Inference


frame_files = sorted(os.listdir(FRAMES_DIR))
print(f" Found {len(frame_files)} frames in '{FRAMES_DIR}'")

for idx, frame_name in enumerate(frame_files):
    frame_path = os.path.join(FRAMES_DIR, frame_name)
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Skipping unreadable frame: {frame_name}")
        continue

    results = model.predict(source=frame, conf=0.45, iou=0.5, save=False, verbose=False)[0]
    masks = results.masks
    boxes = results.boxes
    names = model.names

    if masks is None or boxes is None or len(masks.data) == 0:
        print(f" No objects detected in frame: {frame_name}")
        continue

    frame_annotation = {"frame": frame_name, "objects": []}
    frame_draw = frame.copy()

    for i in range(len(boxes.cls)):
        class_id = int(boxes.cls[i])
        class_name = names[class_id]

        if class_name not in TARGET_CLASSES:
            continue

        if i >= len(masks.data):
            continue

        mask = masks.data[i].cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Ignore small masks
                continue

            polygon = contour[:, 0, :].tolist()
            if len(polygon) < 3:
                continue

            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_draw, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame_draw, class_name, polygon[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            frame_annotation["objects"].append({
                "class": class_name,
                "polygon": polygon
            })

        mask_filename = os.path.join(MASKS_DIR, f"{frame_name.split('.')[0]}_{class_name}_{i}.png")
        cv2.imwrite(mask_filename, mask * 255)

    if frame_annotation["objects"]:
        annotated_path = os.path.join(ANNOTATED_DIR, f"{frame_name.split('.')[0]}_annotated.jpg")
        cv2.imwrite(annotated_path, frame_draw)
        annotations.append(frame_annotation)
    else:
        print(f"⚠️ No valid target objects in frame: {frame_name}")


#STEP 5: Save to JSON


with open(ANNOTATIONS_JSON, "w") as f:
    json.dump(annotations, f, indent=2)

print(f" Completed polygon annotation. Total annotated frames: {len(annotations)}")
