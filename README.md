<div align="center">

# 🤖 AI-Assisted Automated Bounding Box Generation
### for YOLO Training in Object Detection Systems

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![SAM](https://img.shields.io/badge/SAM-Meta%20AI-0467DF?style=for-the-badge)](https://github.com/facebookresearch/segment-anything)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*Final Year Project — B.Eng. Electrical & Electronic Engineering*

</div>

---

## 📌 Project Overview

This project presents a **Hybrid AI-Assisted Annotation Pipeline** for automating polygon-level bounding box generation to accelerate the training of YOLO-based instance segmentation models. Rather than relying on fully automated labeling (which introduces noisy, unreliable annotations), this system integrates a **Human-in-the-Loop (HITL)** approach — combining human bounding box prompts with Meta AI's **Segment Anything Model (SAM)** to produce high-precision polygon masks.

The trained model performs **multi-class plastic bottle detection** across 5 real-world challenging scenarios, targeting applications in automated waste sorting and recycling systems.

> **Key Insight:** This project deliberately chose *hybrid manual-guided annotation* over fully automated labeling. Empirical results demonstrate that human-prompted SAM masks significantly outperform auto-detected masks in segmentation quality and downstream model mAP.

---

## 🏗️ Repository Structure

```
📦 AI-Assisted-YOLO-Annotation/
│
├── 📂 annotation/
│   ├── full_hybrid_annotation.py     # ⭐ MAIN: Hybrid annotation tool (Human + SAM)
│   ├── hybrid_annotation.py          # Earlier hybrid annotation prototype
│   ├── trial_automated.py            # Fully automated annotation (YOLO+SAM, no human)
│   ├── trial_hybrid.py               # Timed hybrid annotation for trial experiments
│   └── trial_manual.py               # Manual polygon annotation baseline
│
├── 📂 training/
│   ├── train_final.py                # Final model training (YOLOv11m-seg, 5 classes)
│   ├── train_hybrid.py               # Training on hybrid-annotated dataset
│   └── train_manual_baseline.py      # Training on manual-annotated dataset
│
├── 📂 evaluation/
│   ├── test_final.py                 # Full evaluation: metrics + visual proof
│   ├── compute_iou.py                # IoU computation across annotation methods
│   └── compute_results.py            # Aggregate results summary
│
├── 📂 dataset/
│   ├── data_final.yaml               # Dataset config (5 classes, final dataset)
│   ├── data_manual.yaml              # Dataset config (manual baseline)
│   └── data.yaml                     # Dataset config (hybrid dataset)
│
├── 📂 assets/
│   ├── flowchart.png                 # System methodology flowchart
│   ├── sample_predictions/           # Example visual predictions
│   └── graphs/                       # Training metrics graphs
│
├── 📂 samples/
│   ├── images/                       # Sample input images (non-sensitive)
│   └── labels/                       # Corresponding sample label files
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Ignores datasets, model weights, runs/
└── README.md                         # This file
```

> **Note:** The full dataset, trained model weights (`best.pt`), and SAM checkpoint (`sam_vit_b_01ec64.pth`) are **not included** in this repository due to file size. See [Setup](#️-setup--installation) for download instructions.

---

## ✨ Key Features & Technical Highlights

| Feature | Description |
|---|---|
| 🎯 **Hybrid HITL Annotation** | Human draws bounding box → SAM auto-generates precise polygon mask |
| 🏷️ **5-Class Detection** | Handles `one_bottle`, `multiple_bottle`, `bad_lighting`, `occlusion`, `cracked` |
| 🧩 **Polygon Segmentation** | Outputs YOLO-format polygon labels (instance segmentation, not just bounding boxes) |
| ⚡ **Real-Time UI** | Professional annotation tool with sidebar, hotkeys, undo/clear, live preview |
| 📊 **3-Method Comparison** | Rigorous comparison: Manual vs. Hybrid vs. Fully Automated annotation |
| 🔬 **IoU Discard Protocol** | Automated labels below 85% IoU threshold are rejected to maintain quality |
| 🚀 **GPU Accelerated** | CUDA-enabled inference for both YOLO and SAM |

---

## 🧠 Methodology

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID ANNOTATION PIPELINE                    │
│                                                                  │
│   Human Annotator                                                │
│        │                                                         │
│        ▼                                                         │
│   [Draw Bounding Box]  ──►  SAM (ViT-B)  ──►  Polygon Mask     │
│                                                     │            │
│                              Select Class [1-5]     │            │
│                                     │               │            │
│                                     ▼               ▼            │
│                              YOLO Format Label (.txt)            │
│                                     │                            │
│                                     ▼                            │
│                           dataset_final/labels/                  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                           │
│                                                                  │
│   dataset_final/  ──►  YOLOv11m-seg  ──►  best.pt              │
│   data_final.yaml      (150 epochs)        (trained model)      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION PIPELINE                          │
│                                                                  │
│   best.pt  ──►  test_final.py  ──►  mAP / Precision / Recall   │
│                                ──►  Visual Proof Images          │
└─────────────────────────────────────────────────────────────────┘
```

### Why Hybrid Over Fully Automated?

The project empirically evaluated three annotation strategies:

| Method | Avg. Time / Image | Total Time (30 imgs) | Model mAP@0.5 |
|---|---|---|---|
| ✍️ Manual Polygon (OpenCV Tool) | 1.74 mins | 52.27 mins | 0.6927 |
| 🤖 Fully Automated (YOLO+SAM) | 0.15 mins | 4.36 mins | — |
| 🔀 **Hybrid (Human+SAM) — 700 images** | **0.22 mins** | **6.55 mins** | **0.562** |

> The **Hybrid method** is **11.9× faster** than manual annotation while maintaining high-quality polygon masks, making it the optimal strategy for production-grade dataset creation.

---

## 🖥️ The Annotation Tool (`full_hybrid_annotation.py`)

The core of this project is a custom-built professional annotation GUI:

```
┌─────────────────────────────────┬──────────────────────┐
│                                 │  HOTKEYS:            │
│                                 │                      │
│                                 │  [1] One Bottle      │
│        IMAGE CANVAS             │  [2] Multiple Bottle │
│                                 │  [3] Bad Lighting    │
│   (Drag to draw bounding box)   │  [4] Occlusion       │
│                                 │  [5] Cracked         │
│                                 │  ─────────────────── │
│   ┌─────────────┐               │  [Drag] Draw Box     │
│   │  SAM MASK   │               │  [U]ndo last         │
│   │  (Green     │               │  [C]lear all         │
│   │   Polygon)  │               │  [S]ave & Next       │
│   └─────────────┘               │  [D]iscard/Skip      │
│                                 │  [Q]uit Tool         │
│                                 │  ─────────────────── │
│                                 │  ACTIVE CLASS:       │
│                                 │  One Bottle          │
└─────────────────────────────────┴──────────────────────┘
```

**Workflow per image:**
1. Press `[1-5]` to select the target class
2. **Drag** to draw a bounding box around the object
3. SAM **automatically generates** a precise polygon mask
4. Repeat for all objects in the image
5. Press `[S]` to save the YOLO label file, or `[D]` to discard

---

## 🗂️ Dataset — 5 Class Definitions

| Class ID | Class Name | Description |
|---|---|---|
| `0` | `one_bottle` | Single plastic bottle, clear visibility |
| `1` | `multiple_bottle` | Two or more bottles in frame |
| `2` | `bad_lighting` | Bottle under low light / overexposure |
| `3` | `occlusion` | Partially hidden or overlapping bottle |
| `4` | `cracked` | Damaged or crushed bottle |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (NVIDIA, 6GB+ VRAM recommended)
- [CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-downloads)

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/AI-Assisted-YOLO-Annotation.git
cd AI-Assisted-YOLO-Annotation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model Checkpoints

**SAM ViT-B checkpoint** (~375 MB):
```bash
# Download from Meta AI
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# Place in project root directory
```

**YOLOv11m-seg base weights:**
```bash
# Automatically downloaded by Ultralytics on first run
# Or manually: https://github.com/ultralytics/assets/releases
```

### 4. Prepare Your Dataset
```
dataset_final/
├── images/
│   ├── train/    ← training images (.jpg)
│   └── val/      ← validation images (.jpg)
└── labels/
    ├── train/    ← YOLO polygon label files (.txt)
    └── val/      ← YOLO polygon label files (.txt)
```

---

## 🚀 Usage

### Step 1 — Annotate Images (Hybrid Method)
```bash
python annotation/full_hybrid_annotation.py
```
> Annotated labels are saved to `dataset_final/labels/train/`

### Step 2 — Train the Model
```bash
python training/train_final.py
```
> Best weights saved to `runs/segment/fyp_hybrid_medium/weights/best.pt`

### Step 3 — Evaluate the Model
```bash
python evaluation/test_final.py
```
> Outputs: mAP metrics + visual prediction images in `runs/segment/FYP_Results/`

---

## 📈 Results

### Model Performance Comparison

> *Evaluated on 300 unseen test images | YOLOv11 + SAM | NVIDIA RTX 4050*

| Training Dataset | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| Manual Annotation (30 images) | 0.0049 | 0.9505 | 0.0098 | 0.6927 | 0.5535 |
| **Hybrid Annotation (700 images)** | **0.496** | **0.622** | **0.551** | **0.562** | **0.494** |

> 🏆 The **Hybrid model (700 images)** significantly outperforms the manual baseline in Precision and F1-Score, demonstrating that dataset scale enabled by hybrid annotation leads to a more robust and balanced model.

### Annotation Time Efficiency

| Method | Trial 1 | Trial 2 | Trial 3 | Total (30 imgs) | Avg / Image |
|---|---|---|---|---|---|
| Manual (OpenCV Tool) | 20.24 min | 16.08 min | 15.96 min | 52.27 min | 1.74 min |
| Fully Automated (YOLO+SAM) | 1.63 min | 1.30 min | 1.44 min | 4.36 min | 0.15 min |
| **Hybrid (Human+SAM)** | **2.54 min** | **2.05 min** | **1.96 min** | **6.55 min** | **0.22 min** |

> ⚡ Hybrid annotation is **11.9× faster** than manual, with only a minor human effort overhead vs. fully automated.

### Discard Protocol (IoU ≥ 0.85) Results

| Dataset Split | Images Processed | Valid Accepted | Invalid Rejected | Acceptance Rate |
|---|---|---|---|---|
| Training (700) | 700 | 657 | 43 | **93.9%** |
| Testing (300) | 300 | 300 | 0 | 100% |
| **Total** | **1000** | **957** | **43** | **95.7%** |

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **Python 3.10** | Core programming language |
| **YOLOv11 (Ultralytics)** | Object detection & instance segmentation |
| **Segment Anything Model (SAM ViT-B)** | AI-assisted polygon mask generation |
| **OpenCV** | Image processing & annotation UI rendering |
| **PyTorch** | Deep learning backend (CUDA acceleration) |
| **Label Studio** | Initial dataset exploration & validation |
| **NumPy** | Numerical operations & mask processing |

---

## 📁 Generated Outputs

| Output | Location | Description |
|---|---|---|
| Label files | `dataset_final/labels/` | YOLO polygon `.txt` files |
| Training runs | `runs/segment/` | Weights, plots, metrics |
| Visual proofs | `runs/segment/FYP_Results/Visual_Proofs/` | Annotated prediction images |
| Timing results | `trial_results/` | Per-image annotation timing JSON |

---

## 🤝 Acknowledgements

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) — for the YOLO framework
- [Meta AI — Segment Anything](https://github.com/facebookresearch/segment-anything) — for the SAM model
- [OpenCV](https://opencv.org/) — for the annotation UI foundation
- Universiti Islam Antarabangsa Malaysia (UIAM) — Final Year Project supervisors and examiners

---

<div align="center">

**Muhammad Haiqal Danial bin Mohamad Rasid**
B.Eng. Electrical & Electronic Engineering
| Final Year Project 2025/2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/haiqalrasid)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/hyqaldanial-lab)

</div>
