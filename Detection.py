import numpy as np


class Detection:
    """
    Represents a single object detection in one frame.

    Attributes:
        bbox    : np.array [x1, y1, x2, y2] — bounding box in pixel coords
        conf    : float — detection confidence score
        cls     : int   — class index
        cls_name: str   — class label string
        feature : np.array — deep appearance embedding (Re-ID vector)
    """

    def __init__(self, bbox, conf, cls, cls_name, feature=None):
        self.bbox     = np.array(bbox, dtype=float)      # [x1, y1, x2, y2]
        self.conf     = float(conf)
        self.cls      = int(cls)
        self.cls_name = str(cls_name)
        self.feature  = np.array(feature, dtype=float) if feature is not None else None

    def to_xywh(self):
        """Convert [x1, y1, x2, y2] → [cx, cy, w, h]."""
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        return np.array([cx, cy, w, h], dtype=float)

    def to_xyxy(self):
        """Return bounding box as [x1, y1, x2, y2]."""
        return self.bbox.copy()

    def __repr__(self):
        return (
            f"Detection(cls={self.cls_name}, conf={self.conf:.2f}, "
            f"bbox={self.bbox.tolist()})"
        )
