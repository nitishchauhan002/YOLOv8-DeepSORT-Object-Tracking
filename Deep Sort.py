import numpy as np
from ultralytics import YOLO

from deep_sort.detection import Detection

class YOLOv8Detector:
    """
    Wraps Ultralytics YOLOv8 and converts raw model output
    into a list of Detection objects compatible with DeepSORT.
    """

    def __init__(self, model_path: str = "weights/yolov8n.pt",
                 conf_thresh: float = 0.4,
                 device: str = "cpu"):
        """
        Args:
            model_path  : Path to .pt weights file.
            conf_thresh : Minimum confidence to keep a detection.
            device      : 'cpu' or 'cuda' / 'cuda:0'.
        """
        self.model       = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.device      = device
        self.class_names = self.model.names   # {0: 'person', 1: 'bicycle', ...}

    def detect(self, frame: np.ndarray) -> list:
        """
        Run detection on a single BGR frame.

        Args:
            frame: np.ndarray (H, W, 3) — OpenCV BGR image.
        Returns:
            List of Detection objects.
        """
        results = self.model(frame, conf=self.conf_thresh,
                             device=self.device, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf            = float(box.conf[0].cpu().numpy())
            cls             = int(box.cls[0].cpu().numpy())
            cls_name        = self.class_names[cls]

            # Lightweight mock appearance embedding (128-D).
            # ── Replace with a real Re-ID CNN for production. ──
            feature = self._extract_feature(frame, [x1, y1, x2, y2])

            detections.append(
                Detection(
                    bbox     = [x1, y1, x2, y2],
                    conf     = conf,
                    cls      = cls,
                    cls_name = cls_name,
                    feature  = feature,
                )
            )

        return detections

    def _extract_feature(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        Crop the detection patch and compute a simple colour histogram
        as a stand-in appearance embedding.

        For better Re-ID accuracy swap this with:
          - OSNet  (torchreid)
          - ResNet50 fine-tuned on Market-1501
          - Any lightweight Re-ID backbone
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return np.zeros(128, dtype=float)

        # 3-channel colour histogram → 128-D vector
        hist = []
        for c in range(3):
            h_c, _ = np.histogram(patch[:, :, c], bins=43, range=(0, 256))
            hist.extend(h_c)
        hist = np.array(hist[:128], dtype=float)
        norm = np.linalg.norm(hist)
        return hist / (norm + 1e-6)
