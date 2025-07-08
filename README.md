# Assignment-Labeller![frame_0001](https://github.com/user-attachments/assets/f2053fad-0ec5-4c5c-86de-945b8c50af78)
![frame_0001 - Copy](https://github.com/user-attachments/assets/211fdee6-9374-42bd-81c3-adca62c7f329)
![frame_0013](https://github.com/user-attachments/assets/57367260-c129-4b7b-8c68-2beb03df7c47)
![frame_0057](https://github.com/user-attachments/assets/c887f61d-8605-464d-96d5-735bbf4bcea4)
# Polygon Annotation Pipeline – YOLOv8 & SAM Integration

## 🔧 Requirements

### Python Packages
Create a virtual environment and install the following:
```bash
pip install -r requirements.txt
```

**`requirements.txt`**:
```txt
opencv-python
numpy
torch
ultralytics
segment-anything
```

### FFmpeg (for video output)
If you want to save the final output as an `.mp4` or `.avi` video, you need FFmpeg installed and added to your system PATH.
- Download from: https://ffmpeg.org/download.html

## 📂 Folder Structure
```
project-root/
├── yolo_polygon_pipeline.py            # YOLO-only polygon annotation (images + JSON)
├── yolo_sam_polygon_annotator.py       # YOLO + SAM integration (image output)
├── yolo_sam_video_output.py            # YOLO + SAM integration (video output)
├── sample_data/
│   └── input_video.mp4               # Sample input video
├── outputs/
    ├── frames/                         # Extracted frames
    ├── annotated_frames/                # Annotated output images
    ├── annotations.json                 # Polygon JSON
    └── annotated_video.avi              # Final video output
```

---

## ▶️ How to Run the Code

### 1. Extract Frames & Run YOLO (Image + JSON)
```bash
python yolo_polygon_pipeline.py
```
> Output: `.json` + polygon-annotated `.jpg` images

### 2. YOLO + SAM with Annotated Images
```bash
python yolo_sam_polygon_annotator.py
```
> Output: Refined polygon images using Segment Anything

### 3. YOLO + SAM with Annotated Video
```bash
python yolo_sam_video_output.py
```
> Output: `outputs/annotated_video.avi`

---

## 📄 Output Formats

### JSON (`annotations.json`):
```json
{
  "frame": "frame_0012.jpg",
  "objects": [
    {
      "class": "car",
      "polygon": [[x1, y1], [x2, y2], ..., [xn, yn]]
    },
    {
      "class": "person",
      "polygon": [[x1, y1], [x2, y2], ..., [xn, yn]]
    }
  ]
}
```

### Video:
- `.avi` format (requires FFmpeg-compatible codec)
- Can be easily converted to `.mp4` using online tools or FFmpeg

---

## 🎓 Tips
- Tested on low-end CPU machine with `yolov8n` and `vit_b` SAM variant
- Video should be short for testing (under 30s recommended)
- You can easily adjust target classes from the `TARGET_CLASSES` list in code

---

## 🚀 What's Inside
- Automatic annotation using YOLO + SAM
- Polygons drawn tightly around object shapes (not just boxes)
- Easy integration into labeling tools

---

## 🛫 Sample Use Case
Want to label traffic scenes with precise polygons? This repo gives you a plug-and-play toolchain for that.

---

For questions, improvements, or collaborations, feel free to reach out!
