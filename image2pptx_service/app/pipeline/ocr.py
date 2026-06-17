from __future__ import annotations

import os
from pathlib import Path

from app.config import settings
from app.schemas import OcrItem


class OcrEngine:
    name = 'base'
    def detect(self, image_path: str) -> list[OcrItem]:
        raise NotImplementedError


class DummyOcrEngine(OcrEngine):
    name = 'dummy'
    def detect(self, image_path: str) -> list[OcrItem]: return []


class TesseractOcrEngine(OcrEngine):
    name = 'tesseract'
    def detect(self, image_path: str) -> list[OcrItem]:
        import pytesseract
        from PIL import Image
        data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
        items=[]
        for i, text in enumerate(data.get('text', [])):
            text=(text or '').strip()
            conf=float(data.get('conf', ['-1'])[i] or -1)
            if text and conf >= 0:
                items.append(OcrItem(text=text, bbox_px=[data['left'][i], data['top'][i], data['width'][i], data['height'][i]], confidence=conf/100.0))
        return items


DEFAULT_PADDLEOCR_MODEL_DIRS = {
    'det_model_dir': ('ch_PP-OCRv4_det_infer', 'PADDLEOCR_DET_MODEL_DIR'),
    'rec_model_dir': ('ch_PP-OCRv4_rec_infer', 'PADDLEOCR_REC_MODEL_DIR'),
    'cls_model_dir': ('ch_ppocr_mobile_v2.0_cls_infer', 'PADDLEOCR_CLS_MODEL_DIR'),
}


def _existing_dir(path: Path, label: str) -> str:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f'{label} points to a missing PaddleOCR model directory: {path}')
    return str(path)


def _repo_model_base_dir() -> Path:
    return settings.storage_dir / 'models' / 'paddleocr'


def paddleocr_kwargs_from_local_models(require_complete: bool = True) -> dict[str, str]:
    """Read PaddleOCR model dirs from the fixed project-local storage path.

    Expected layout:
      storage/models/paddleocr/ch_PP-OCRv4_det_infer
      storage/models/paddleocr/ch_PP-OCRv4_rec_infer
      storage/models/paddleocr/ch_ppocr_mobile_v2.0_cls_infer
    """
    base_dir = _repo_model_base_dir()
    candidates = {
        kwarg: base_dir / dirname
        for kwarg, (dirname, _env_name) in DEFAULT_PADDLEOCR_MODEL_DIRS.items()
    }
    existing = {kwarg: str(path) for kwarg, path in candidates.items() if path.exists()}
    if not existing:
        if require_complete:
            expected = ', '.join(str(path) for path in candidates.values())
            raise FileNotFoundError(f'Project-local PaddleOCR models are missing. Expected: {expected}. Set PADDLEOCR_ALLOW_DOWNLOAD=1 to let PaddleOCR download models automatically.')
        return {}
    if len(existing) != len(candidates):
        missing = ', '.join(str(path) for path in candidates.values() if not path.exists())
        raise FileNotFoundError(f'Incomplete project-local PaddleOCR model set. Missing: {missing}')
    return existing


def paddleocr_kwargs_from_env() -> dict[str, str]:
    """Read optional PaddleOCR model directory overrides from environment variables."""
    kwargs = {}
    for kwarg, (_dirname, env_name) in DEFAULT_PADDLEOCR_MODEL_DIRS.items():
        value = os.getenv(env_name)
        if value:
            kwargs[kwarg] = _existing_dir(Path(value), env_name)
    return kwargs


def paddleocr_model_kwargs() -> dict[str, str]:
    """Return PaddleOCR model dirs: env overrides first, then project-local dirs.

    By default this refuses to return empty kwargs, because empty kwargs make
    PaddleOCR download models during initialization. Set PADDLEOCR_ALLOW_DOWNLOAD=1
    only in environments where first-run downloads are acceptable.
    """
    env_kwargs = paddleocr_kwargs_from_env()
    if env_kwargs:
        return env_kwargs
    if os.getenv('PADDLEOCR_ALLOW_DOWNLOAD') == '1':
        return paddleocr_kwargs_from_local_models(require_complete=False)
    return paddleocr_kwargs_from_local_models(require_complete=True)


def create_paddleocr():
    """Create PaddleOCR with project-local/offline model settings applied."""
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang='ch', **paddleocr_model_kwargs())


class PaddleOcrEngine(OcrEngine):
    name = 'paddleocr'
    def __init__(self):
        self.ocr = create_paddleocr()
    def detect(self, image_path: str) -> list[OcrItem]:
        result = self.ocr.ocr(image_path, cls=True) or []
        items=[]
        for page in result:
            for line in page or []:
                pts, (text, conf) = line
                xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
                items.append(OcrItem(text=text, bbox_px=[min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)], confidence=float(conf)))
        return items


def get_ocr_engine(prefer: str = 'auto') -> tuple[OcrEngine, list[str]]:
    warnings=[]
    if prefer in ('auto','paddle'):
        try: return PaddleOcrEngine(), warnings
        except Exception as e: warnings.append(f'PaddleOCR unavailable: {e.__class__.__name__}: {e}')
    if prefer in ('auto','tesseract'):
        try: return TesseractOcrEngine(), warnings
        except Exception as e: warnings.append(f'Tesseract unavailable: {e.__class__.__name__}: {e}')
    warnings.append('Using Dummy OCR engine; no native text boxes may be created.')
    return DummyOcrEngine(), warnings
