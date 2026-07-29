from ultralytics import YOLO

print("="*55)
print("  TRAINING — Manual Annotation Baseline (30 images)")
print("="*55)

model = YOLO('yolo11n-seg.pt')   # Start from pretrained base

if __name__ == '__main__':
    # Train on the 30 manually annotated images
    results = model.train(
        data='data_manual.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        patience=20,          # Stop early if no improvement
        project='runs/segment/Hasil_FYP',
        name='Model_Manual_30',
        exist_ok=True,
        verbose=True
    )

    print("\n" + "="*55)
    print("  EVALUATION — Testing on 300 Unseen Images")
    print("="*55)

    # Load the best weights from this manual training
    trained = YOLO('runs/segment/Hasil_FYP/Model_Manual_30/weights/best.pt')

    metrics = trained.val(
        data='data_manual.yaml',
        split='val',
        project='FYP_Results',
        name='Manual_Testing_Metrics'
    )

    print("\n===== FINAL TABLE 5 RESULTS (Manual Baseline) =====")
    print(f"Precision  : {metrics.box.mp:.3f}")
    print(f"Recall     : {metrics.box.mr:.3f}")
    print(f"mAP50      : {metrics.box.map50:.3f}")
    print(f"mAP50-95   : {metrics.box.map:.3f}")
    print(f"Mask mAP50 : {metrics.seg.map50:.3f}")
    print("="*55)
