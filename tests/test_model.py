from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/v8/yolov8-p2-BiFPN.yaml")
# model.info()
model.train(
    data="coco8.yaml",
    epochs=3,
    imgsz=640,
    batch=4,
    device='cpu',       # or 'cpu'
    workers=0,      # important on Windows — avoids DataLoader multiprocessing issues
    plots=False,
)