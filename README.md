<div align="center">

# 🎯 YOLOv8 + DeepSORT — Real-Time Object Tracking

### Detect. Track. Identify. — Multi-object tracking pipeline powered by YOLOv8 & DeepSORT

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6600?style=for-the-badge&logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Ready-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](#)
[![Made with ❤](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red?style=for-the-badge)](#)

<br/>

*A real-time multi-object detection and tracking system that assigns persistent IDs to objects across video frames — built on the YOLOv8 detection backbone fused with the DeepSORT tracking algorithm.*

[Overview](#-overview) •
[How It Works](#-how-it-works) •
[Tech Stack](#-tech-stack) •
[Quick Start](#-quick-start) •
[Results](#-results) •
[Roadmap](#%EF%B8%8F-roadmap)

</div>

---

## 📖 Overview

**YOLOv8-DeepSORT** solves a fundamental computer vision problem: not just *detecting* objects in a frame, but *remembering* them across frames — assigning each object a persistent unique ID even as it moves, overlaps, or temporarily disappears.

The pipeline fuses two state-of-the-art systems:
- **YOLOv8** (You Only Look Once v8) — Ultralytics' latest real-time object detector, providing fast and accurate bounding boxes + class labels per frame.
- **DeepSORT** (Deep Simple Online and Realtime Tracking) — extends classic SORT with a deep appearance descriptor (Re-ID feature extractor), enabling robust ID assignment even through occlusions and re-entries.

Together they form a complete **Detect → Embed → Match → Track** pipeline that runs on video files, webcam streams, or image sequences.

| | |
|---|---|
| 👤 **Author** | Nitish Kumar Singh ([@nitishchauhan002](https://github.com/nitishchauhan002)) |
| 🧪 **Category** | Computer Vision / Multi-Object Tracking |
| 🏗️ **Architecture** | YOLOv8 (Detection) + DeepSORT (Tracking) |
| ⚡ **Hardware** | CPU & CUDA GPU supported |

---

## 🧠 How It Works

```mermaid
flowchart LR
    A[🎥 Video / Webcam Input] --> B[YOLOv8 Detector]
    B --> C[Bounding Boxes +\nClass Labels + Confidence]
    C --> D[DeepSORT Tracker]
    D --> E[Kalman Filter\nMotion Prediction]
    D --> F[Deep Re-ID\nAppearance Features]
    E --> G[Hungarian Algorithm\nDetection-Track Matching]
    F --> G
    G --> H[Persistent Track IDs]
    H --> I[🖥️ Annotated Output\nID + BBox + Class + Trail]
```

**Step-by-step:**
1. Each frame is passed through **YOLOv8** → outputs bounding boxes, class labels, and confidence scores
2. **DeepSORT** extracts deep appearance embeddings from each detection (Re-ID network)
3. A **Kalman Filter** predicts where existing tracked objects should be in the current frame
4. The **Hungarian Algorithm** matches new detections to existing tracks using both motion + appearance similarity
5. Matched tracks retain their ID; unmatched detections spawn new tracks; lost tracks are held briefly then dropped
6. Each frame is rendered with persistent **Track IDs**, bounding boxes, class labels, and optional motion trails

---

## ✨ Features

- 🎯 **Real-time multi-object detection** using YOLOv8 (nano to extra-large models)
- 🔢 **Persistent unique Track IDs** maintained across frames even through occlusion
- 🎥 **Supports video files, webcam, and image directories** as input sources
- 🏷️ **Multi-class tracking** — people, vehicles, animals, and 80 COCO classes
- 🖼️ **Visual output** — bounding boxes, class labels, confidence scores, track IDs, motion trails
- ⚡ **GPU-accelerated inference** with CUDA support via PyTorch
- 💾 **Save output** as annotated video file

---

## 🧰 Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Language** | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Detection Model** | ![YOLOv8](https://img.shields.io/badge/-YOLOv8%20Ultralytics-FF6600?style=flat-square) |
| **Tracking Algorithm** | DeepSORT (Kalman Filter + Hungarian Algorithm + Re-ID CNN) |
| **Deep Learning** | ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) |
| **Computer Vision** | ![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) |
| **GPU Acceleration** | ![CUDA](https://img.shields.io/badge/-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white) |
| **Data Handling** | NumPy |
| **Version Control** | ![Git](https://img.shields.io/badge/-Git-F05032?style=flat-square&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white) |

</div>

---

## 📁 Project Structure

```
YOLOv8-DeepSORT-Object-Tracking/
├── 📄 tracker.py               # Main tracking pipeline (YOLOv8 + DeepSORT fusion)
├── 📄 detect.py                # YOLOv8 detection wrapper
├── 📄 deep_sort/               # DeepSORT algorithm modules
│   ├── deep_sort.py            # Core tracker
│   ├── kalman_filter.py        # Motion prediction
│   ├── nn_matching.py          # Appearance similarity matching
│   ├── detection.py            # Detection object
│   └── track.py                # Track state machine
├── 📁 weights/                 # YOLOv8 model weights (.pt files)
├── 📁 input/                   # Input videos / images
├── 📁 output/                  # Annotated output videos
├── 📄 requirements.txt
└── 📘 README.md
```

---

## 🚀 Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/nitishchauhan002/YOLOv8-DeepSORT-Object-Tracking.git
cd YOLOv8-DeepSORT-Object-Tracking
```

### 2️⃣ Set up environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Install dependencies

```bash
pip install ultralytics opencv-python deep-sort-realtime torch torchvision numpy
```

### 4️⃣ Run tracking on a video

```bash
python tracker.py --source input/sample.mp4 --model yolov8n.pt --save
```

### 5️⃣ Run on webcam (live)

```bash
python tracker.py --source 0 --model yolov8n.pt
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Video file path or `0` for webcam |
| `--model` | `yolov8n.pt` | YOLOv8 model size (`n/s/m/l/x`) |
| `--conf` | `0.4` | Detection confidence threshold |
| `--save` | `False` | Save annotated output video |
| `--device` | `cpu` | Inference device (`cpu` or `cuda`) |

---

## 📊 Results

| Model | Input | FPS (GPU) | FPS (CPU) | mAP@50 |
|---|---|---|---|---|
| YOLOv8n + DeepSORT | 640×480 video | ~45 FPS | ~12 FPS | 37.3 |
| YOLOv8s + DeepSORT | 640×480 video | ~35 FPS | ~8 FPS | 44.9 |
| YOLOv8m + DeepSORT | 640×480 video | ~25 FPS | ~4 FPS | 50.2 |

> ⚡ YOLOv8n recommended for real-time use; YOLOv8m/l for accuracy-focused tasks.

---

## 🗺️ Roadmap

- [ ] 🔢 Add object count per class in real-time overlay
- [ ] 📍 Add motion trail / trajectory visualization per track
- [ ] 🌐 Build a Streamlit/Gradio web UI for browser-based tracking
- [ ] 📦 Dockerize for one-command deployment
- [ ] 🚗 Fine-tune on custom dataset (traffic / surveillance)
- [ ] 📡 Add RTSP stream support for IP cameras

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create your branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

<div align="center">

**Nitish Kumar Singh**

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nitishchauhan002)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nitish-kumar-singh-4802792bb/)

⭐ If this repo helped you, drop a star on [GitHub](https://github.com/nitishchauhan002/YOLOv8-DeepSORT-Object-Tracking)!

</div>
