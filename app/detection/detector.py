import torch
from ultralytics import YOLO
import numpy as np

class TorchDetector:
    def __init__(self, model_path="models/best.pt", use_gpu=False):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.class_names = list(self.model.names.values())
        self.model.to(self.device)
        self.conf_default = 0.25
        self.conf_no_class = 0.75

    def detect(self, frame: np.ndarray) -> list:
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            threshold = self.conf_no_class if cls_name.startswith("NO-") else self.conf_default
            if conf < threshold:
                continue
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": cls_id,
                "class_name": self.model.names[cls_id],
                "confidence": conf
            })
        return detections

