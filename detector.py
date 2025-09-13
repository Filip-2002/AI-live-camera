import time
import cv2
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = YOLO(cfg.get('model_path', 'yolov8n.pt'))
        self.img_size = cfg.get('img_size', 640)
        self.conf = cfg.get('conf_threshold', 0.4)
        self.iou = cfg.get('iou_threshold', 0.5)
        self.max_det = cfg.get('max_det', 100)
        self.classes = cfg.get('classes', None)
        self.names = self.model.model.names if hasattr(self.model.model, 'names') else self.model.names

    def infer(self, frame):
        t0 = time.time()
        results = self.model.predict(source=frame, imgsz=self.img_size, conf=self.conf,
                                     iou=self.iou, max_det=self.max_det, classes=self.classes,
                                     verbose=False)
        dt = time.time() - t0
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                xyxy = b.xyxy.cpu().numpy().astype(int)[0]
                conf = float(b.conf.cpu().numpy()[0])
                cls = int(b.cls.cpu().numpy()[0])
                dets.append([*xyxy, conf, cls])
        return dets, dt

    def label(self, cls_id):
        if isinstance(self.names, dict):
            return self.names.get(cls_id, str(cls_id))
        return self.names[cls_id] if cls_id < len(self.names) else str(cls_id)
