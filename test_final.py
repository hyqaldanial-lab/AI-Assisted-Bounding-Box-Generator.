"""
Run testing on the 300 validation images using the best trained YOLOv11 model.
Step 1: Compute metrics (mAP, Precision, Recall) on CPU to avoid HD mask OOM.
Step 2: Save visual predictions on GPU with images pre-resized to 1280px.
"""
from ultralytics import YOLO
import os, glob, cv2
import torch

BEST_MODEL = "runs/segment/runs/segment/fyp_hybrid_medium/weights/best.pt"
VAL_IMAGES = "dataset_final/images/val"
VISUAL_DIR = "runs/segment/FYP_Results/Visual_Proofs"
os.makedirs(VISUAL_DIR, exist_ok=True)

CLASS_NAMES = {0: "One Bottle", 1: "Multiple Bottle",
               2: "Bad Lighting", 3: "Occlusion", 4: "Cracked"}
COLORS = {0: (0,255,0), 1:(255,100,0), 2:(0,255,255),
          3:(0,165,255), 4:(0,0,255)}

if __name__ == '__main__':

    # ── STEP 1: Metrics on CPU ────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Computing Validation Metrics on CPU")
    print("=" * 60)

    model_cpu = YOLO(BEST_MODEL)
    metrics = model_cpu.val(
        data     = "data_final.yaml",
        split    = "val",
        imgsz    = 640,
        batch    = 1,
        device   = "cpu",
        workers  = 0,
        plots    = True,
        project  = "runs/segment/FYP_Results",
        name     = "Testing_Metrics",
        exist_ok = True,
    )

    print("\n=== FINAL METRICS ===")
    print(f"  Box  Precision  : {metrics.box.mp:.4f}")
    print(f"  Box  Recall     : {metrics.box.mr:.4f}")
    print(f"  Box  mAP@0.5   : {metrics.box.map50:.4f}")
    print(f"  Box  mAP@.5:.95: {metrics.box.map:.4f}")
    print(f"  Mask Precision  : {metrics.seg.mp:.4f}")
    print(f"  Mask Recall     : {metrics.seg.mr:.4f}")
    print(f"  Mask mAP@0.5   : {metrics.seg.map50:.4f}")
    print(f"  Mask mAP@.5:.95: {metrics.seg.map:.4f}")

    # Free CPU model
    del model_cpu
    torch.cuda.empty_cache()

    # ── STEP 2: Visual predictions on GPU ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Saving Visual Predictions on GPU")
    print("=" * 60)

    model_gpu = YOLO(BEST_MODEL)

    images = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg")))
    print(f"Processing {len(images)} images...\n")

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize HD images to max 1280px on longest side
        h, w = img.shape[:2]
        scale = min(1280 / w, 1280 / h, 1.0)
        img_small = cv2.resize(img, (int(w * scale), int(h * scale))) \
                    if scale < 1.0 else img.copy()

        with torch.no_grad():
            results = model_gpu.predict(
                img_small, imgsz=640, device="cuda",
                verbose=False, conf=0.25)

        annotated = img_small.copy()

        for result in results:
            if result.masks is None:
                continue
            for mask, box in zip(result.masks.data, result.boxes):
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                color  = COLORS.get(cls_id, (255,255,255))
                label  = f"{CLASS_NAMES.get(cls_id,'?')} {conf:.2f}"

                # Mask overlay
                m = cv2.resize(mask.cpu().numpy(),
                               (img_small.shape[1], img_small.shape[0]))
                overlay = annotated.copy()
                overlay[m > 0.5] = color
                annotated = cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0)

                # Box + label
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
                cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
                cv2.putText(annotated, label, (x1+2, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        out = os.path.join(VISUAL_DIR, os.path.basename(img_path))
        cv2.imwrite(out, annotated)

        if idx % 30 == 0 or idx == len(images):
            print(f"  [{idx}/{len(images)}] saved...")

    print(f"\nAll Done!")
    print(f"  Visual proofs : {VISUAL_DIR}")
    print(f"  Metrics plots : runs/segment/FYP_Results/Testing_Metrics/")
