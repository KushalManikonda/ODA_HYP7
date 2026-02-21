import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


class MarineDetector:
    def __init__(self, weight_path="models/marine_detector_ruod.pth", score_threshold=0.35):
        self.weight_path = weight_path
        self.score_threshold = score_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label_to_name = {}

        if os.path.exists(weight_path):
            self._load_model(weight_path)

    def _load_model(self, weight_path):
        checkpoint = torch.load(weight_path, map_location=self.device)
        num_classes = int(checkpoint.get("num_classes", 11))

        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        raw_map = checkpoint.get("label_to_name", {})
        self.label_to_name = {int(k): v for k, v in raw_map.items()}
        self.model = model

    def is_ready(self):
        return self.model is not None

    def predict(self, image_bgr: np.ndarray):
        if self.model is None:
            return []

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(Image.fromarray(image_rgb)).to(self.device)

        with torch.no_grad():
            outputs = self.model([tensor])[0]

        boxes = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()

        results = []
        for box, label, score in zip(boxes, labels, scores):
            if float(score) < self.score_threshold:
                continue
            x1, y1, x2, y2 = box.astype(int).tolist()
            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "label_id": int(label),
                    "label_name": self.label_to_name.get(int(label), f"class_{int(label)}"),
                    "score": float(score),
                }
            )

        return results
