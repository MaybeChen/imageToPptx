from pathlib import Path
import json
import logging
import os
import platform
import re
import struct
import subprocess
import tempfile
import textwrap
from app.config import settings
from app.schemas import SegmentItem


logger = logging.getLogger(__name__)


def _segment_log(message: str) -> None:
    text = f'[segment] {message}'
    print(text, flush=True)
    logger.info(text)


def _extract_windows_dll_path(message: str) -> Path | None:
    match = re.search(r'Error loading "([^"]+)"', message, flags=re.IGNORECASE)
    if not match:
        return None
    return Path(match.group(1))


def _read_pe_imports(dll_path: Path) -> list[str]:
    data = dll_path.read_bytes()
    if len(data) < 0x40 or data[:2] != b'MZ':
        return []
    pe_offset = struct.unpack_from('<I', data, 0x3C)[0]
    if pe_offset + 248 > len(data) or data[pe_offset:pe_offset + 4] != b'PE\0\0':
        return []
    number_of_sections = struct.unpack_from('<H', data, pe_offset + 6)[0]
    optional_header_size = struct.unpack_from('<H', data, pe_offset + 20)[0]
    optional_header_offset = pe_offset + 24
    magic = struct.unpack_from('<H', data, optional_header_offset)[0]
    data_directory_offset = optional_header_offset + (112 if magic == 0x20B else 96)
    import_rva, import_size = struct.unpack_from('<II', data, data_directory_offset + 8)
    if not import_rva or not import_size:
        return []
    section_offset = optional_header_offset + optional_header_size
    sections: list[tuple[int, int, int, int]] = []
    for index in range(number_of_sections):
        offset = section_offset + index * 40
        virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from('<IIII', data, offset + 8)
        sections.append((virtual_address, max(virtual_size, raw_size), raw_pointer, raw_size))

    def rva_to_offset(rva: int) -> int | None:
        for virtual_address, virtual_size, raw_pointer, raw_size in sections:
            if virtual_address <= rva < virtual_address + virtual_size:
                file_offset = raw_pointer + (rva - virtual_address)
                if file_offset < raw_pointer + raw_size and file_offset < len(data):
                    return file_offset
        return None

    imports: list[str] = []
    descriptor_offset = rva_to_offset(import_rva)
    if descriptor_offset is None:
        return []
    while descriptor_offset + 20 <= len(data):
        original_first_thunk, _time, _forwarder, name_rva, first_thunk = struct.unpack_from(
            '<IIIII', data, descriptor_offset
        )
        if not any((original_first_thunk, name_rva, first_thunk)):
            break
        name_offset = rva_to_offset(name_rva)
        if name_offset is not None:
            end = data.find(b'\0', name_offset)
            if end != -1:
                imports.append(data[name_offset:end].decode('ascii', errors='replace'))
        descriptor_offset += 20
    return imports


def _windows_dll_search_dirs(dll_path: Path) -> list[Path]:
    dirs = [dll_path.parent]
    windir = os.environ.get('WINDIR', r'C:\Windows')
    dirs.extend([Path(windir) / 'System32', Path(windir) / 'SysWOW64', Path(windir)])
    dirs.extend(Path(path) for path in os.environ.get('PATH', '').split(os.pathsep) if path)
    seen: set[str] = set()
    unique_dirs: list[Path] = []
    for directory in dirs:
        key = str(directory).lower()
        if key not in seen:
            seen.add(key)
            unique_dirs.append(directory)
    return unique_dirs


def _find_windows_dll(name: str, search_dirs: list[Path]) -> Path | None:
    for directory in search_dirs:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _diagnose_windows_torch_dll_error(exc: Exception) -> str:
    dll_path = _extract_windows_dll_path(str(exc))
    if not dll_path:
        return 'No DLL path was found in the loader message, so Windows did not expose the missing dependency name.'
    if not dll_path.exists():
        return f'Loader target does not exist: {dll_path}'
    try:
        imports = _read_pe_imports(dll_path)
    except Exception as diagnostic_exc:  # noqa: BLE001 - diagnostics must not mask the original load error
        return f'Could not inspect {dll_path} imports: {diagnostic_exc.__class__.__name__}: {diagnostic_exc}'
    if not imports:
        return f'No direct imported DLLs were found in {dll_path}; the missing dependency may be delay-loaded or transitive.'
    search_dirs = _windows_dll_search_dirs(dll_path)
    missing = [name for name in imports if _find_windows_dll(name, search_dirs) is None]
    if missing:
        return f'Missing direct DLL dependencies for {dll_path.name}: {", ".join(missing)}'
    python_bits = platform.architecture()[0]
    return (
        f'All direct DLL dependencies for {dll_path.name} were found in the current DLL search path. '
        f'The remaining cause is likely a transitive/delay-loaded dependency, architecture mismatch, '
        f'or incompatible binary version (current Python: {python_bits}).'
    )

def _format_yolo_exception(exc: Exception) -> str:
    detail = f'{exc.__class__.__name__}: {exc}'
    message = str(exc).lower()
    is_windows_torch_dll_error = (
        os.name == 'nt'
        and isinstance(exc, OSError)
        and ('winerror 126' in message or 'winerror 127' in message)
        and ('torch' in message or 'shm.dll' in message or 'fbgemm.dll' in message)
    )
    if is_windows_torch_dll_error:
        diagnostic = _diagnose_windows_torch_dll_error(exc)
        return (
            f'{detail}. Windows PyTorch DLL dependency load failed; YOLO was selected but cannot start. Diagnostic: {diagnostic}. '
            'The named DLL file can exist while one of its dependent DLLs is missing or ABI-incompatible. '
            'Install/repair Microsoft Visual C++ Redistributable 2015-2022. If you already have another Python environment where YOLO works, '
            'set YOLO_PYTHON to that python.exe; otherwise reinstall a CPU PyTorch wheel inside this Poetry environment: '
            'poetry run python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu'
        )
    if isinstance(exc, ModuleNotFoundError) and getattr(exc, 'name', '') in ('ultralytics', 'sahi'):
        return f'{detail}. Install YOLO/SAHI dependencies in this environment with poetry install, or set YOLO_PYTHON to a Python executable where SAHI+YOLO inference already works'
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = (exc.stderr or '').strip()
        stdout = (exc.stdout or '').strip()
        subprocess_detail = '; '.join(part for part in (f'stdout={stdout}' if stdout else '', f'stderr={stderr}' if stderr else '') if part)
        if subprocess_detail:
            return f'{detail}. YOLO_PYTHON subprocess failed: {subprocess_detail}'
    return detail


SEGMENT_LAYER_PRIORITY = {
    'background': 0,
    'shape': 1,
    'image': 2,
    'icon': 2,
    'chart': 2,
    'table': 2,
    'line': 3,
    'arrow': 3,
}


YOLO_CLASS_ID_LABELS = {
    '0': 'icon',
    '1': 'logo',
    '2': 'image',
    '3': 'chart',
    '4': 'table',
    '5': 'rectangle',
    '6': 'rounded_rectangle',
    '7': 'circle',
    '8': 'ellipse',
    '9': 'diamond',
    '10': 'line',
    '11': 'arrow',
    '12': 'background',
    '13': 'panel',
    '14': 'container',
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
    'table': 'table',
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


def yolo_model_dirs() -> list[Path]:
    """Return supported project-local YOLO model directories in priority order.

    Historically the service documented storage/models/yolo. Some deployments
    mount models/yolo at the repository root, so support both without requiring
    an environment variable.
    """
    repo_root = settings.base_dir.parent
    dirs = [
        settings.storage_dir / 'models' / 'yolo',
        settings.base_dir / 'models' / 'yolo',
        repo_root / 'models' / 'yolo',
    ]
    unique_dirs: list[Path] = []
    for directory in dirs:
        if directory not in unique_dirs:
            unique_dirs.append(directory)
    return unique_dirs


def yolo_model_dir() -> Path:
    return yolo_model_dirs()[0]


def _iter_yolo_model_candidates() -> list[Path]:
    candidates: list[Path] = []
    for model_dir in yolo_model_dirs():
        for pattern in ('best.pt', '*.pt', '*.onnx', '*.engine'):
            candidates.extend(model_dir.glob(pattern))
    return sorted(set(candidates), key=lambda path: (path.name != 'best.pt', str(path)))


def has_yolo_model() -> bool:
    env_path = os.getenv('YOLO_MODEL_PATH')
    if env_path:
        return Path(env_path).expanduser().exists()
    return bool(_iter_yolo_model_candidates())


def find_yolo_model_path() -> Path:
    env_path = os.getenv('YOLO_MODEL_PATH')
    if env_path:
        path = Path(env_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f'YOLO_MODEL_PATH points to a missing model file: {path}')
        return path

    candidates = _iter_yolo_model_candidates()
    if not candidates:
        searched = ', '.join(str(path) for path in yolo_model_dirs())
        raise FileNotFoundError(f'YOLO model file is missing. Put best.pt or another YOLO model under one of: {searched}; or set YOLO_MODEL_PATH.')
    return candidates[0]


def _normalize_yolo_label(label: str) -> str:
    return label.lower().replace('-', '_').strip()


def _label_to_segment_type(label: str) -> str | None:
    normalized = _normalize_yolo_label(label)
    normalized = YOLO_CLASS_ID_LABELS.get(normalized, normalized)
    if normalized in YOLO_IGNORED_LABELS or normalized.replace('_', ' ') in YOLO_IGNORED_LABELS:
        return None
    return YOLO_LABEL_TYPE_MAP.get(normalized, YOLO_LABEL_TYPE_MAP.get(normalized.replace('_', ' '), 'image'))


def _label_to_shape(label: str) -> str | None:
    normalized = _normalize_yolo_label(label)
    normalized = YOLO_CLASS_ID_LABELS.get(normalized, normalized)
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


def _detection_label(detection: dict) -> str:
    class_id = detection.get('class_id')
    if class_id is not None:
        mapped = YOLO_CLASS_ID_LABELS.get(str(class_id))
        if mapped:
            return mapped
    return str(detection.get('label', ''))


def _detections_to_segments(detections: list[dict]) -> list[SegmentItem]:
    items: list[SegmentItem] = []
    for detection in detections:
        label = _detection_label(detection)
        seg_type = _label_to_segment_type(label)
        if seg_type is None:
            continue
        bbox = _xyxy_to_bbox(detection.get('xyxy', [0, 0, 0, 0]))
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        items.append(SegmentItem(
            type=seg_type,
            shape=_label_to_shape(label),
            bbox_px=bbox,
            confidence=_value_to_float(detection.get('confidence', 0.0)),
        ))
    return merge_segments_by_layer(items)


_SUBPROCESS_YOLO_SCRIPT = """
import json
import sys
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def object_prediction_to_detection(prediction):
    bbox = getattr(prediction, 'bbox', None)
    category = getattr(prediction, 'category', None)
    score = getattr(prediction, 'score', None)
    minx = getattr(bbox, 'minx', 0.0)
    miny = getattr(bbox, 'miny', 0.0)
    maxx = getattr(bbox, 'maxx', 0.0)
    maxy = getattr(bbox, 'maxy', 0.0)
    category_id = getattr(category, 'id', None)
    category_name = getattr(category, 'name', category_id)
    score_value = getattr(score, 'value', score if score is not None else 0.0)
    return {
        'class_id': category_id,
        'label': str(category_name),
        'confidence': float(score_value),
        'xyxy': [float(minx), float(miny), float(maxx), float(maxy)],
    }


image_path, model_path, output_path = sys.argv[1:4]
conf = float(sys.argv[4])
iou = float(sys.argv[5])
imgsz = int(sys.argv[6])
max_det = int(sys.argv[7])
slice_height = int(sys.argv[8])
slice_width = int(sys.argv[9])
overlap_height_ratio = float(sys.argv[10])
overlap_width_ratio = float(sys.argv[11])
device = sys.argv[12]
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=model_path,
    confidence_threshold=conf,
    device=device,
)
result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    postprocess_match_metric='IOU',
    postprocess_match_threshold=iou,
    perform_standard_pred=False,
    verbose=0,
)
detections = [object_prediction_to_detection(prediction) for prediction in result.object_prediction_list[:max_det]]
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(detections, f)
"""



def _color_for_yolo_type(label: str) -> str:
    seg_type = _label_to_segment_type(label) or 'image'
    return {
        'background': '#9CA3AF',
        'shape': '#2563EB',
        'image': '#F97316',
        'icon': '#A855F7',
        'chart': '#EF4444',
        'table': '#14B8A6',
        'line': '#22C55E',
        'arrow': '#84CC16',
    }.get(seg_type, '#F59E0B')


def _draw_debug_box(draw, font, xyxy, label: str, color: str) -> None:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    text_bbox = draw.textbbox((x1, y1), label, font=font)
    label_h = text_bbox[3] - text_bbox[1]
    label_w = text_bbox[2] - text_bbox[0]
    label_y = max(0, y1 - label_h - 4)
    draw.rectangle((x1, label_y, x1 + label_w + 6, label_y + label_h + 4), fill=color)
    draw.text((x1 + 3, label_y + 2), label, fill='white', font=font)


def write_yolo_detection_debug_overlay(image_path: Path, detections: list[dict], output_path: Path, title: str = 'YOLO raw') -> Path:
    from PIL import Image, ImageDraw, ImageFont

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path).convert('RGB') as image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for index, detection in enumerate(detections, 1):
            label = _detection_label(detection)
            class_id = detection.get('class_id')
            display_label = f'{index}:{class_id}:{label}' if class_id is not None else f'{index}:{label}'
            confidence = _value_to_float(detection.get('confidence', 0.0))
            display_label = f'{display_label} {confidence:.2f}'
            _draw_debug_box(draw, font, detection.get('xyxy', [0, 0, 0, 0]), display_label, _color_for_yolo_type(label))
        draw.text((8, 8), f'{title}: {len(detections)} raw boxes', fill='#111827', font=font)
        image.save(output_path)
    _segment_log(f'{title} debug overlay saved: {output_path}')
    return output_path


def write_segment_debug_overlay(image_path: Path, segments: list[SegmentItem], output_path: Path, title: str = 'segments') -> Path:
    detections = []
    for segment in segments:
        x, y, w, h = [float(value) for value in segment.bbox_px]
        detections.append({
            'label': segment.type,
            'confidence': segment.confidence,
            'xyxy': [x, y, x + w, y + h],
        })
    return write_yolo_detection_debug_overlay(image_path, detections, output_path, title)


def _sahi_prediction_to_detection(prediction) -> dict:
    bbox = getattr(prediction, 'bbox', None)
    category = getattr(prediction, 'category', None)
    score = getattr(prediction, 'score', None)
    category_id = getattr(category, 'id', None)
    category_name = getattr(category, 'name', category_id)
    score_value = getattr(score, 'value', score if score is not None else 0.0)
    return {
        'class_id': category_id,
        'label': str(category_name),
        'confidence': _value_to_float(score_value),
        'xyxy': [
            _value_to_float(getattr(bbox, 'minx', 0.0)),
            _value_to_float(getattr(bbox, 'miny', 0.0)),
            _value_to_float(getattr(bbox, 'maxx', 0.0)),
            _value_to_float(getattr(bbox, 'maxy', 0.0)),
        ],
    }


def _sahi_settings() -> dict:
    return {
        'slice_height': _env_int('SAHI_SLICE_HEIGHT', 512),
        'slice_width': _env_int('SAHI_SLICE_WIDTH', 512),
        'overlap_height_ratio': _env_float('SAHI_OVERLAP_HEIGHT_RATIO', 0.2),
        'overlap_width_ratio': _env_float('SAHI_OVERLAP_WIDTH_RATIO', 0.2),
        'device': os.getenv('SAHI_DEVICE', 'cpu'),
    }


def detect_segments_with_yolo_subprocess(image_path: Path, model_path: Path, debug_image_path: Path | None = None) -> list[SegmentItem]:
    yolo_python = os.getenv('YOLO_PYTHON')
    if not yolo_python:
        raise RuntimeError('YOLO_PYTHON is not set')
    conf = _env_float('YOLO_CONF', 0.01)
    iou = _env_float('YOLO_IOU', 0.7)
    imgsz = _env_int('YOLO_IMGSZ', 1024)
    max_det = _env_int('YOLO_MAX_DET', 80)
    sahi_settings = _sahi_settings()
    timeout = _env_int('YOLO_SUBPROCESS_TIMEOUT', 120)
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as output:
        output_path = Path(output.name)
    try:
        _segment_log(f'YOLO subprocess start: python={yolo_python} model={model_path} image={image_path}')
        completed = subprocess.run(
            [
                yolo_python,
                '-c',
                textwrap.dedent(_SUBPROCESS_YOLO_SCRIPT),
                str(image_path),
                str(model_path),
                str(output_path),
                str(conf),
                str(iou),
                str(imgsz),
                str(max_det),
                str(sahi_settings['slice_height']),
                str(sahi_settings['slice_width']),
                str(sahi_settings['overlap_height_ratio']),
                str(sahi_settings['overlap_width_ratio']),
                sahi_settings['device'],
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.stdout.strip():
            _segment_log(f'YOLO subprocess stdout: {completed.stdout.strip()}')
        if completed.stderr.strip():
            _segment_log(f'YOLO subprocess stderr: {completed.stderr.strip()}')
        detections = json.loads(output_path.read_text(encoding='utf-8'))
        if debug_image_path:
            write_yolo_detection_debug_overlay(image_path, detections, debug_image_path)
        merged = _detections_to_segments(detections)
        _segment_log(f'YOLO subprocess done: raw_items={len(detections)} merged_items={len(merged)}')
        if debug_image_path:
            write_segment_debug_overlay(image_path, merged, debug_image_path, 'YOLO')
        return merged
    finally:
        output_path.unlink(missing_ok=True)


def detect_segments_with_yolo(image_path: Path, debug_image_path: Path | None = None) -> list[SegmentItem]:
    model_path = find_yolo_model_path()
    if os.getenv('YOLO_PYTHON'):
        return detect_segments_with_yolo_subprocess(image_path, model_path, debug_image_path)

    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    conf = _env_float('YOLO_CONF', 0.01)
    iou = _env_float('YOLO_IOU', 0.7)
    max_det = _env_int('YOLO_MAX_DET', 80)
    sahi_settings = _sahi_settings()
    _segment_log(
        f"YOLO SAHI enabled: loading model={model_path} image={image_path} "
        f"conf={conf} iou={iou} max_det={max_det} "
        f"slice={sahi_settings['slice_width']}x{sahi_settings['slice_height']} "
        f"overlap={sahi_settings['overlap_width_ratio']}x{sahi_settings['overlap_height_ratio']} "
        f"device={sahi_settings['device']}"
    )
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=str(model_path),
        confidence_threshold=conf,
        device=sahi_settings['device'],
    )
    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=sahi_settings['slice_height'],
        slice_width=sahi_settings['slice_width'],
        overlap_height_ratio=sahi_settings['overlap_height_ratio'],
        overlap_width_ratio=sahi_settings['overlap_width_ratio'],
        postprocess_match_metric='IOU',
        postprocess_match_threshold=iou,
        perform_standard_pred=False,
        verbose=0,
    )
    detections = [
        _sahi_prediction_to_detection(prediction)
        for prediction in getattr(result, 'object_prediction_list', [])[:max_det]
    ]
    if debug_image_path:
        write_yolo_detection_debug_overlay(image_path, detections, debug_image_path)
    merged = _detections_to_segments(detections)
    _segment_log(f'YOLO predict done: raw_items={len(detections)} merged_items={len(merged)}')
    if debug_image_path:
        write_segment_debug_overlay(image_path, merged, debug_image_path, 'YOLO')
    return merged


def detect_segments_with_opencv(image_path: Path) -> list[SegmentItem]:
    import cv2

    _segment_log(f'OpenCV segmentation start: image={image_path}')
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
    result = items[:30]
    _segment_log(f'OpenCV segmentation done: items={len(result)}')
    return result


def detect_segments(image_path: Path, mode: str = 'balanced', debug_image_path: Path | None = None) -> list[SegmentItem]:
    if mode == 'fast':
        _segment_log(f'Segmentation skipped: mode=fast image={image_path}')
        return []
    engine = os.getenv('SEGMENT_ENGINE', 'auto').lower()
    _segment_log(f'Segmentation dispatch: engine={engine} mode={mode} image={image_path}')
    try:
        if engine in ('yolo', 'yolo11', 'yolo26'):
            _segment_log(f'Segmentation selected: YOLO forced by SEGMENT_ENGINE={engine}')
            return (
                detect_segments_with_yolo(image_path, debug_image_path)
                if debug_image_path
                else detect_segments_with_yolo(image_path)
            )[:30]
        if engine == 'opencv':
            _segment_log('Segmentation selected: OpenCV forced by SEGMENT_ENGINE=opencv')
            return detect_segments_with_opencv(image_path)
        if engine == 'auto':
            if has_yolo_model():
                _segment_log('Segmentation selected: auto detected YOLO model, trying YOLO first')
                try:
                    yolo_items = (
                        detect_segments_with_yolo(image_path, debug_image_path)
                        if debug_image_path
                        else detect_segments_with_yolo(image_path)
                    )
                    if yolo_items:
                        _segment_log(f'Segmentation result: using YOLO items={len(yolo_items[:30])}')
                        return yolo_items[:30]
                    _segment_log('Segmentation fallback: YOLO returned 0 items, using OpenCV')
                except Exception as exc:
                    _segment_log(f'Segmentation fallback: YOLO failed with {_format_yolo_exception(exc)}; using OpenCV')
            else:
                _segment_log('Segmentation selected: auto found no YOLO model, using OpenCV')
        else:
            _segment_log(f'Segmentation selected: unknown engine={engine}, using OpenCV')
        result = detect_segments_with_opencv(image_path)
        _segment_log(f'Segmentation result: using OpenCV items={len(result)}')
        return result
    except Exception as exc:
        _segment_log(f'Segmentation failed: {_format_yolo_exception(exc)}')
        return []
