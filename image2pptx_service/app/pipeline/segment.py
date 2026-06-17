from pathlib import Path
from app.schemas import SegmentItem


SEGMENT_LAYER_PRIORITY = {
    'background': 0,
    'shape': 1,
    'image': 2,
    'icon': 2,
    'chart': 2,
    'line': 3,
    'arrow': 3,
}


def _area(bbox) -> float:
    return max(0, bbox[2]) * max(0, bbox[3])


def _intersection_area(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def _coverage(inner, outer) -> float:
    return _intersection_area(inner, outer) / max(1, _area(inner))


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


def _classify_contour(img, contour, slide_area) -> SegmentItem | None:
    import cv2

    x, y, cw, ch = cv2.boundingRect(contour)
    area = cw * ch
    if area < slide_area * 0.0003 or area > slide_area * 0.75:
        return None

    bbox = [x, y, cw, ch]
    aspect = cw / max(1, ch)
    approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
    contour_area = max(1.0, float(cv2.contourArea(contour)))
    rect_fill = contour_area / max(1, area)

    # Edit-Banana style: route elements into dedicated semantic groups instead
    # of letting one generic detector decide everything. These are lightweight
    # prompt-group substitutes for the local OpenCV adapter.
    if (cw >= 32 and ch <= 8 and aspect >= 4) or (ch >= 32 and cw <= 8 and aspect <= 0.25):
        return SegmentItem(type='line', bbox_px=bbox, confidence=0.62)

    if area <= slide_area * 0.015 and 0.5 <= aspect <= 2.0 and not _looks_like_simple_shape(img, bbox):
        return SegmentItem(type='icon', bbox_px=bbox, confidence=0.62)

    if len(approx) >= 4 and cw > 20 and ch > 20 and rect_fill > 0.45 and _looks_like_simple_shape(img, bbox):
        shape = 'rect'
        if len(approx) > 6 and 0.75 <= aspect <= 1.35:
            shape = 'ellipse'
        return SegmentItem(type='shape', shape=shape, bbox_px=bbox, confidence=0.72)

    if cw > 24 and ch > 16:
        return SegmentItem(type='image', bbox_px=bbox, confidence=0.58)
    return None


def merge_segments_by_layer(items: list[SegmentItem]) -> list[SegmentItem]:
    """Deduplicate overlapping candidates with Edit-Banana-like layer priority."""
    ordered = sorted(
        items,
        key=lambda item: (
            SEGMENT_LAYER_PRIORITY.get(item.type, 9),
            -float(item.confidence),
            -_area(item.bbox_px),
        ),
    )
    kept: list[SegmentItem] = []
    for item in ordered:
        duplicate = False
        for existing in kept:
            if _coverage(item.bbox_px, existing.bbox_px) >= 0.85:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def detect_segments(image_path: Path, mode: str = 'balanced') -> list[SegmentItem]:
    if mode == 'fast':
        return []
    try:
        import cv2

        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items = []
        slide_area = w * h
        for contour in contours:
            item = _classify_contour(img, contour, slide_area)
            if item is not None:
                items.append(item)
        items = merge_segments_by_layer(items)
        if not items:
            # Keep a conservative visual asset fallback, while full-slide background
            # fallback is handled in manifest.py.
            items.append(SegmentItem(type='image', bbox_px=[w * 0.08, h * 0.16, w * 0.84, h * 0.68], confidence=0.35))
        return items[:30]
    except Exception:
        return []
