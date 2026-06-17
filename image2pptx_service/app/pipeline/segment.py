from pathlib import Path
from app.schemas import SegmentItem


def _looks_like_simple_shape(img, bbox) -> bool:
    """Heuristic: native shape only when the region is visually simple.

    Complex cards with icons/text/gradients should be kept as image assets so
    the visual result remains close to the source screenshot.
    """
    try:
        import cv2
        import numpy as np

        x, y, w, h = [int(v) for v in bbox]
        roi = img[max(0, y): max(0, y + h), max(0, x): max(0, x + w)]
        if roi.size == 0:
            return False
        color_std = float(np.mean(np.std(roi.reshape(-1, 3), axis=0)))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edge_density = float(cv2.countNonZero(cv2.Canny(gray, 50, 150))) / max(1, w * h)
        return color_std < 22 and edge_density < 0.08
    except Exception:
        return False


def detect_segments(image_path: Path, mode: str = 'balanced') -> list[SegmentItem]:
    if mode == 'fast':
        return []
    try:
        import cv2

        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < (w * h) * 0.003 or area > (w * h) * 0.75:
                continue
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
            bbox = [x, y, cw, ch]
            if len(approx) >= 4 and cw > 20 and ch > 20 and _looks_like_simple_shape(img, bbox):
                items.append(SegmentItem(type='shape', shape='rect', bbox_px=bbox, confidence=0.7))
            elif cw > 24 and ch > 16:
                items.append(SegmentItem(type='image', bbox_px=bbox, confidence=0.6))
        if not items:
            # Keep a conservative visual asset fallback, while full-slide background
            # fallback is handled in manifest.py.
            items.append(SegmentItem(type='image', bbox_px=[w * 0.08, h * 0.16, w * 0.84, h * 0.68], confidence=0.35))
        return items[:30]
    except Exception:
        return []
