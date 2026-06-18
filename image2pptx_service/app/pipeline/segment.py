from pathlib import Path
import os
from app.config import settings
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

YOLO_LABEL_TYPE_MAP = {
    'icon': 'icon',
    'logo': 'icon',
    'image': 'image',
    'picture': 'image',
    'photo': 'image',
    'chart': 'chart',
    'diagram': 'chart',
    'graph': 'chart',
    'table': 'chart',
    'shape': 'shape',
    'rectangle': 'shape',
    'rounded_rectangle': 'shape',
    'rounded rectangle': 'shape',
    'circle': 'shape',
    'ellipse': 'shape',
    'diamond': 'shape',
    'line': 'line',
    'connector': 'line',
    'arrow': 'arrow',
    'background': 'background',
    'panel': 'background',
    'container': 'background',
}

YOLO_IGNORED_LABELS = {'text', 'text_region', 'text region', 'ocr', 'word'}

YOLO_SHAPE_LABELS = {
    'rectangle': 'rect',
    'rounded_rectangle': 'rounded_rect',
    'rounded rectangle': 'rounded_rect',
    'circle': 'ellipse',
    'ellipse': 'ellipse',
    'diamond': 'diamond',
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


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
            if existing.type == 'background' and item.type != 'background':
                continue
            if item.type == 'background' and existing.type != 'background':
                continue
            if _coverage(item.bbox_px, existing.bbox_px) >= 0.85:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def yolo_model_dir() -> Path:
    return settings.storage_dir / 'models' / 'yolo'


def find_yolo_model_path() -> Path:
    env_path = os.getenv('YOLO_MODEL_PATH')
    if env_path:
        path = Path(env_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f'YOLO_MODEL_PATH points to a missing model file: {path}')
        return path

    model_dir = yolo_model_dir()
    candidates = []
    for pattern in ('*.pt', '*.onnx', '*.engine'):
        candidates.extend(model_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f'YOLO model file is missing. Put a YOLO11 model under {model_dir} or set YOLO_MODEL_PATH.')
    return sorted(candidates)[0]


def _normalize_yolo_label(label: str) -> str:
    return label.lower().replace('-', '_').strip()


def _label_to_segment_type(label: str) -> str | None:
    normalized = _normalize_yolo_label(label)
    if normalized in YOLO_IGNORED_LABELS or normalized.replace('_', ' ') in YOLO_IGNORED_LABELS:
        return None
    return YOLO_LABEL_TYPE_MAP.get(normalized, YOLO_LABEL_TYPE_MAP.get(normalized.replace('_', ' '), 'image'))


def _label_to_shape(label: str) -> str | None:
    normalized = _normalize_yolo_label(label)
    return YOLO_SHAPE_LABELS.get(normalized) or YOLO_SHAPE_LABELS.get(normalized.replace('_', ' '))


def _xyxy_to_bbox(xyxy) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _value_to_float(value, default: float = 0.0) -> float:
    try:
        if hasattr(value, 'item'):
            return float(value.item())
        return float(value)
    except Exception:
        return default


def detect_segments_with_yolo(image_path: Path) -> list[SegmentItem]:
    from ultralytics import YOLO

    model_path = find_yolo_model_path()
    model = YOLO(str(model_path))
    conf = _env_float('YOLO_CONF', 0.25)
    iou = _env_float('YOLO_IOU', 0.7)
    imgsz = _env_int('YOLO_IMGSZ', 1024)
    max_det = _env_int('YOLO_MAX_DET', 80)
    results = model.predict(str(image_path), conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False)
    names = getattr(model, 'names', {}) or {}
    items: list[SegmentItem] = []
    for result in results or []:
        result_names = getattr(result, 'names', None) or names
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(_value_to_float(getattr(box, 'cls', 0)))
            label = str(result_names.get(cls_id, cls_id)) if isinstance(result_names, dict) else str(cls_id)
            seg_type = _label_to_segment_type(label)
            if seg_type is None:
                continue
            confidence = _value_to_float(getattr(box, 'conf', 0.0))
            bbox = _xyxy_to_bbox(getattr(box, 'xyxy')[0] if hasattr(getattr(box, 'xyxy'), '__getitem__') else getattr(box, 'xyxy'))
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            items.append(SegmentItem(type=seg_type, shape=_label_to_shape(label), bbox_px=bbox, confidence=confidence))
    return merge_segments_by_layer(items)


def detect_segments_with_opencv(image_path: Path) -> list[SegmentItem]:
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


def detect_segments(image_path: Path, mode: str = 'balanced') -> list[SegmentItem]:
    if mode == 'fast':
        return []
    engine = os.getenv('SEGMENT_ENGINE', 'opencv').lower()
    try:
        if engine in ('yolo', 'yolo11', 'yolo26'):
            return detect_segments_with_yolo(image_path)[:30]
        return detect_segments_with_opencv(image_path)
    except Exception:
        return []
