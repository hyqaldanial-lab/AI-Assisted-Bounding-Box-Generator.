import cv2
import numpy as np
import os
import glob
import torch
from segment_anything import sam_model_registry, SamPredictor

# ── CONFIGURATION ──────────────────────────────────────────────────────────
INPUT_DIR  = "dataset_final/images/train"
LABEL_DIR  = "dataset_final/labels/train"
os.makedirs(LABEL_DIR, exist_ok=True)

CLASSES = {
    0: {"name": "One Bottle",      "color": (0, 255, 0),   "key": ord('1')},
    1: {"name": "Multiple Bottle", "color": (255, 0, 0),   "key": ord('2')},
    2: {"name": "Bad Lighting",    "color": (0, 255, 255), "key": ord('3')},
    3: {"name": "Occlusion",       "color": (0, 165, 255), "key": ord('4')},
    4: {"name": "Cracked",         "color": (0, 0, 255),   "key": ord('5')}
}
active_class_id = 0

print("Loading SAM (ViT-B) ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device=device)
predictor = SamPredictor(sam)
print("SAM loaded successfully!\n")

# UI State Variables
SIDEBAR_WIDTH = 400
drawing = False
ix, iy = -1, -1
temp_box = None
polygons = []   
view_w, view_h, canvas_h = 0, 0, 0
canvas_base = None  # Pre-rendered canvas to eliminate lag!

def build_sidebar(height):
    """Draws a solid sidebar menu."""
    sidebar = np.zeros((height, SIDEBAR_WIDTH, 3), dtype=np.uint8)
    
    cv2.putText(sidebar, "HOTKEYS:", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    y = 90
    for cid, cinfo in CLASSES.items():
        text = f"[{cid+1}] {cinfo['name']}"
        cv2.putText(sidebar, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cinfo["color"], 2)
        y += 35
        
    y += 10
    cv2.putText(sidebar, "-----------------", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 35
    cv2.putText(sidebar, "[Drag] Draw Box", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    y += 35
    cv2.putText(sidebar, "[U]ndo last", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    y += 35
    cv2.putText(sidebar, "[C]lear all", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    y += 35
    cv2.putText(sidebar, "[S]ave & Next", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 35
    cv2.putText(sidebar, "[D]iscard/Skip", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y += 35
    cv2.putText(sidebar, "[Q]uit Tool", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    y += 60
    active_name = CLASSES[active_class_id]["name"]
    active_color = CLASSES[active_class_id]["color"]
    cv2.putText(sidebar, "ACTIVE CLASS:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(sidebar, active_name, (20, y+40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, active_color, 3)
    
    return sidebar

def update_canvas_base(view_img):
    """Pre-renders the background and sidebar to eliminate mouse drag lag."""
    global canvas_base
    canvas_base = np.zeros((canvas_h, view_w + SIDEBAR_WIDTH, 3), dtype=np.uint8)
    canvas_base[:view_h, :view_w] = view_img
    canvas_base[:, view_w:] = build_sidebar(canvas_h)

def render():
    """Lightning fast render using pre-built canvas_base."""
    display_img = canvas_base.copy()
    
    # Draw completed polygons
    for poly in polygons:
        cv2.drawContours(display_img, [poly['points']], -1, poly['color'], 3)
        
    # Draw dragging box
    if drawing and temp_box is not None:
        x1, y1, x2, y2 = temp_box
        cv2.rectangle(display_img, (x1, y1), (x2, y2), CLASSES[active_class_id]["color"], 2)
        
    cv2.imshow(win_name, display_img)

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, temp_box, polygons
    
    if x > view_w or y > view_h:
        return
        
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        temp_box = [x, y, x, y]
        render()
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_box = [min(ix, x), min(iy, y), max(ix, x), max(iy, y)]
            render()
            
    elif event == cv2.EVENT_LBUTTONUP:
        if not drawing: return
        drawing = False
        
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        temp_box = None
        
        if x2 - x1 > 5 and y2 - y1 > 5:
            # We now pass the HIGH-RES view box directly to SAM!
            human_box = np.array([x1, y1, x2, y2])
            masks, _, _ = predictor.predict(box=human_box[None, :], multimask_output=True)
            
            best_mask, max_area = None, 0
            for i in range(masks.shape[0]):
                area = np.sum(masks[i] > 0)
                if area > max_area:
                    max_area, best_mask = area, masks[i]
            
            if best_mask is not None:
                mask_u8 = (best_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    epsilon = 0.002 * cv2.arcLength(largest, True)
                    approx = cv2.approxPolyDP(largest, epsilon, True)
                    
                    polygons.append({
                        'class_id': active_class_id,
                        'points': approx,
                        'color': CLASSES[active_class_id]["color"]
                    })
        render()

# ── MAIN LOOP ──────────────────────────────────────────────────────────────
image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))

win_name = "Professional Hybrid Annotation Tool"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback(win_name, mouse_callback)

saved_count = 0

for idx, img_path in enumerate(image_paths, 1):
    filename = os.path.basename(img_path)
    name_only = os.path.splitext(filename)[0]
    label_path = os.path.join(LABEL_DIR, f"{name_only}.txt")
    
    if os.path.exists(label_path):
        continue
        
    print(f"\n--- Image {idx}/{len(image_paths)}: {filename} ---")
    
    orig_img = cv2.imread(img_path)
    if orig_img is None: continue
        
    orig_h, orig_w = orig_img.shape[:2]
    
    scale = min(960.0 / orig_h, 1400.0 / orig_w)
    view_h = int(orig_h * scale)
    view_w = int(orig_w * scale)
    canvas_h = max(view_h, 700)
    
    # Upscale the image using cubic interpolation for smoothness
    view_img = cv2.resize(orig_img, (view_w, view_h), interpolation=cv2.INTER_CUBIC)
    
    # SAM now processes the HIGH-RES view image, resulting in smooth, high-quality masks!
    image_rgb = cv2.cvtColor(view_img, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    
    update_canvas_base(view_img)
    polygons = []
    render()
    
    done = False
    while not done:
        key = cv2.waitKey(20) & 0xFF
        
        # Change Class
        if key == CLASSES[0]["key"]: active_class_id = 0; update_canvas_base(view_img); render()
        elif key == CLASSES[1]["key"]: active_class_id = 1; update_canvas_base(view_img); render()
        elif key == CLASSES[2]["key"]: active_class_id = 2; update_canvas_base(view_img); render()
        elif key == CLASSES[3]["key"]: active_class_id = 3; update_canvas_base(view_img); render()
        elif key == CLASSES[4]["key"]: active_class_id = 4; update_canvas_base(view_img); render()
            
        elif key == ord('u'):
            if polygons: polygons.pop(); render()
        elif key == ord('c'):
            polygons = []; render()
            
        elif key == ord('s') or key == 13: 
            if not polygons:
                print("Warning: No labels drawn! Press 'd' if you want to skip without saving.")
                continue
            
            yolo_lines = []
            for poly in polygons:
                line = [str(poly['class_id'])]
                for pt in poly['points']:
                    # Normalize based on the HIGH-RES view dimensions
                    nx = pt[0][0] / view_w
                    ny = pt[0][1] / view_h
                    nx = max(0.0, min(1.0, nx))
                    ny = max(0.0, min(1.0, ny))
                    line.append(f"{nx:.6f}")
                    line.append(f"{ny:.6f}")
                yolo_lines.append(" ".join(line))
            
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
            saved_count += 1
            print(f"Saved {len(polygons)} labels for {filename}.")
            done = True
            
        elif key == ord('d') or key == 32: 
            print(f"Skipped {filename}.")
            done = True
            
        elif key == ord('q') or key == 27: 
            print("\nTool stopped by user.")
            cv2.destroyAllWindows()
            exit(0)

cv2.destroyAllWindows()
print(f"\nAll Done! Saved {saved_count} new images.")
