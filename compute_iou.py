"""
IoU Accuracy Calculator
=======================
Compares Automated and Hybrid polygon labels against Manual ground-truth labels.

Folder layout expected:
  trial_manual_labels/   ← YOLO polygon .txt files from Roboflow manual annotation
  trial_auto_labels/     ← output of trial_automated.py
  trial_hybrid_labels/   ← output of trial_hybrid.py
  trial_results/         ← timing JSONs live here; accuracy JSON written here

Output:
  trial_results/accuracy_results.json   – per-image + aggregate IoU scores
  trial_results/summary_table.json      – clean table for the report / dashboard

Run:  python compute_iou.py
"""

import os
import glob
import json
import numpy as np
import cv2

# ── CONFIG ──────────────────────────────────────────────────────────────────
BATCHES     = [1, 2, 3]   # compute for all three batches
RESULTS_DIR = "trial_results"
IMAGE_DIR   = "30_testing_images"   # for image dimensions
# Labels live in subfolders: trial_manual_labels/batch1/, etc.
MANUAL_BASE = "trial_manual_labels"
AUTO_BASE   = "trial_automated_labels"
HYBRID_BASE = "trial_hybrid_labels"
# ────────────────────────────────────────────────────────────────────────────


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_yolo_polygons(label_path):
    """Return list of numpy arrays [(N,2)] of absolute pixel coords.
       Requires the image dimensions for de-normalisation."""
    polys = []
    if not os.path.exists(label_path):
        return polys
    with open(label_path) as f:
        for line in f.read().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            coords = list(map(float, parts[1:]))
            pts    = [(coords[i], coords[i+1]) for i in range(0, len(coords)-1, 2)]
            polys.append(np.array(pts, dtype=np.float32))
    return polys


def poly_to_mask(poly, h, w):
    """Rasterise a normalised polygon (values 0-1) into a binary mask."""
    abs_pts = (poly * np.array([w, h])).astype(np.int32)
    mask    = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [abs_pts], 1)
    return mask


def polygon_iou(polyA, polyB, h, w):
    """IoU between two normalised polygons via mask rasterisation."""
    mA = poly_to_mask(polyA, h, w)
    mB = poly_to_mask(polyB, h, w)
    inter = np.logical_and(mA, mB).sum()
    union = np.logical_or(mA, mB).sum()
    return inter / union if union > 0 else 0.0


def best_match_iou(ref_polys, cmp_polys, h, w):
    """
    For each reference polygon find the best-matching comparison polygon.
    Returns the average best-match IoU, and the number of unmatched refs.
    """
    if not ref_polys:
        return None, 0
    if not cmp_polys:
        return 0.0, len(ref_polys)

    ious, unmatched = [], 0
    for ref in ref_polys:
        best = max(polygon_iou(ref, cmp, h, w) for cmp in cmp_polys)
        ious.append(best)
        if best == 0:
            unmatched += 1
    return float(np.mean(ious)), unmatched


# ── Main computation ─────────────────────────────────────────────────────────────────────
all_auto_iou, all_hybrid_iou = [], []
batch_results = []   # one entry per batch

os.makedirs(RESULTS_DIR, exist_ok=True)

for batch in BATCHES:
    manual_dir  = os.path.join(MANUAL_BASE, f"batch{batch}")
    auto_dir    = os.path.join(AUTO_BASE,   f"batch{batch}")
    hybrid_dir  = os.path.join(HYBRID_BASE, f"batch{batch}")
    batch_img_dir = f"trial_batch_{batch}"

    manual_files = sorted(glob.glob(os.path.join(manual_dir, "*.txt")))
    if not manual_files:
        print(f"[WARN] Batch {batch}: no manual labels in '{manual_dir}' - skipping.")
        continue

    print(f"\n--- Batch {batch} ({len(manual_files)} images) ---")
    print(f"  {'Image':<53} {'Auto IoU':>9} {'Hybrid IoU':>11}")
    print("  " + "-" * 76)

    per_image, b_auto, b_hybrid = [], [], []

    for man_path in manual_files:
        filename  = os.path.basename(man_path)
        name_only = os.path.splitext(filename)[0]

        # Find image in batch folder first, fall back to main folder
        img_path = os.path.join(batch_img_dir, name_only + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGE_DIR, name_only + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] Cannot read image for {filename} - skipping.")
            continue
        h, w = img.shape[:2]

        man_polys    = read_yolo_polygons(man_path)
        auto_polys   = read_yolo_polygons(os.path.join(auto_dir,   filename))
        hybrid_polys = read_yolo_polygons(os.path.join(hybrid_dir, filename))

        auto_iou,   _ = best_match_iou(man_polys, auto_polys,   h, w)
        hybrid_iou, _ = best_match_iou(man_polys, hybrid_polys, h, w)

        row = {
            "image":        filename,
            "auto_iou":     round(auto_iou,   4) if auto_iou   is not None else None,
            "hybrid_iou":   round(hybrid_iou, 4) if hybrid_iou is not None else None,
        }
        per_image.append(row)
        if auto_iou   is not None: b_auto.append(auto_iou)
        if hybrid_iou is not None: b_hybrid.append(hybrid_iou)

        a_str = f"{auto_iou:.4f}"   if auto_iou   is not None else "  N/A  "
        h_str = f"{hybrid_iou:.4f}" if hybrid_iou is not None else "  N/A  "
        print(f"  {filename:<55} {a_str:>9} {h_str:>11}")

    avg_auto_b   = float(np.mean(b_auto))   if b_auto   else 0.0
    avg_hybrid_b = float(np.mean(b_hybrid)) if b_hybrid else 0.0
    print(f"  {'Batch avg':<55} {avg_auto_b:.4f}    {avg_hybrid_b:.4f}")

    # Load timing for this batch
    def load_json(fname):
        p = os.path.join(RESULTS_DIR, fname)
        return json.load(open(p)) if os.path.exists(p) else {}

    mt = load_json(f"manual_batch{batch}_timing.json")
    at = load_json(f"auto_batch{batch}_timing.json")
    ht = load_json(f"hybrid_batch{batch}_timing.json")

    batch_results.append({
        "batch":            batch,
        "num_images":       len(per_image),
        "manual_time_min":  mt.get("total_time_min"),
        "auto_time_min":    at.get("total_time_min"),
        "hybrid_time_min":  ht.get("total_time_min") or ht.get("total_wall_time_min"),
        "auto_avg_iou":     round(avg_auto_b,   4),
        "hybrid_avg_iou":   round(avg_hybrid_b, 4),
        "per_image":        per_image,
    })
    all_auto_iou.extend(b_auto)
    all_hybrid_iou.extend(b_hybrid)

# -- Overall averages ---------------------------------------------------------------------
avg_auto   = float(np.mean(all_auto_iou))   if all_auto_iou   else 0.0
avg_hybrid = float(np.mean(all_hybrid_iou)) if all_hybrid_iou else 0.0

print("\n" + "=" * 78)
print(f"  Overall Avg Auto   IoU (30 images): {avg_auto:.4f}  ({avg_auto*100:.1f}%)")
print(f"  Overall Avg Hybrid IoU (30 images): {avg_hybrid:.4f}  ({avg_hybrid*100:.1f}%)")
print(f"  Manual (baseline)                 : 1.0000  (100.0%)")
print("=" * 78)

# -- Save accuracy results -----------------------------------------------------
acc_path = os.path.join(RESULTS_DIR, "accuracy_results.json")
with open(acc_path, "w") as f:
    json.dump({
        "overall_auto_avg_iou":   round(avg_auto,   4),
        "overall_hybrid_avg_iou": round(avg_hybrid, 4),
        "manual_avg_iou":         1.0,
        "batches":                batch_results,
    }, f, indent=2)
print(f"  [OK] Accuracy data -> {acc_path}")

# -- Build Table 3 summary (rows = methods, cols = Trial 1/2/3) ---------------
table3 = {
    "description": "Table 3: Time Efficiency Comparison (10 images per trial)",
    "rows": []
}
for method_key, method_name, time_key in [
    ("manual",  "Manual (Polygon)",           "manual_time_min"),
    ("auto",    "Fully Automated (YOLO+SAM)", "auto_time_min"),
    ("hybrid",  "Hybrid (YOLO+SAM)",          "hybrid_time_min"),
]:
    row = {"method": method_name, "trials": []}
    for br in batch_results:
        iou_key = f"{method_key}_avg_iou" if method_key != "manual" else None
        row["trials"].append({
            "trial":    br["batch"],
            "time_min": br.get(time_key),
            "avg_iou":  br.get(iou_key, 1.0) if iou_key else 1.0,
        })
    table3["rows"].append(row)

# Also include a flat "methods" list for the existing dashboard
table3["methods"] = [
    {
        "name":           "Manual (Polygon)",
        "avg_iou":        1.0,
        "avg_iou_pct":    100.0,
        "total_time_min": sum(br["manual_time_min"] for br in batch_results
                             if br.get("manual_time_min")) or None,
        "avg_time_s":     None,
        "note":           "Ground-truth baseline",
    },
    {
        "name":           "Fully Automated (YOLO+SAM)",
        "avg_iou":        round(avg_auto, 4),
        "avg_iou_pct":    round(avg_auto * 100, 1),
        "total_time_min": sum(br["auto_time_min"] for br in batch_results
                             if br.get("auto_time_min")) or None,
        "avg_time_s":     None,
        "note":           "No human input",
    },
    {
        "name":           "Hybrid (Human-in-the-Loop)",
        "avg_iou":        round(avg_hybrid, 4),
        "avg_iou_pct":    round(avg_hybrid * 100, 1),
        "total_time_min": sum(br["hybrid_time_min"] for br in batch_results
                             if br.get("hybrid_time_min")) or None,
        "avg_time_s":     None,
        "note":           "Human bounding box + SAM polygon",
    },
]

sum_path = os.path.join(RESULTS_DIR, "summary_table.json")
with open(sum_path, "w") as f:
    json.dump(table3, f, indent=2)
print(f"  [OK] Table 3 summary -> {sum_path}")
print("\n  DONE! Open 'results_dashboard.html' to view your results.")
