"""
Retrain with YOLOv11s-seg (small model - better accuracy than nano).
"""
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11m-seg.pt")   # medium model - best accuracy for FYP

    results = model.train(
        data     = "data_final.yaml",
        epochs   = 150,              # more epochs for better convergence
        imgsz    = 640,
        batch    = 2,                # small batch for 6GB VRAM with medium model
        device   = "cuda",
        workers  = 0,                # Windows fix
        project  = "runs/segment",
        name     = "fyp_hybrid_medium",
        patience = 30,               # more patience for bigger model
        save     = True,
        plots    = True,
        verbose  = True,
        # Augmentation to help with bad_lighting and occlusion classes
        hsv_h    = 0.015,
        hsv_s    = 0.7,
        hsv_v    = 0.4,
        fliplr   = 0.5,
        mosaic   = 1.0,
        degrees  = 5.0,
    )

    print("\nTraining complete!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
