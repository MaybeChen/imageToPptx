from __future__ import annotations

import os
from pathlib import Path

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


def _existing_dir_from_env(name: str) -> str | None:
    value = os.getenv(name)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'{name} points to a missing PaddleOCR model directory: {path}')
    return str(path)


def paddleocr_kwargs_from_env() -> dict[str, str]:
    """Read local PaddleOCR model directories from environment variables.

    PaddleOCR downloads model tarballs during initialization when model dirs are
    not supplied. Corporate Windows environments may block that HTTPS download
    with a self-signed-certificate error, so these env vars let callers point to
    pre-downloaded/extracted models and avoid network access.
    """
    mapping = {
        'det_model_dir': 'PADDLEOCR_DET_MODEL_DIR',
        'rec_model_dir': 'PADDLEOCR_REC_MODEL_DIR',
        'cls_model_dir': 'PADDLEOCR_CLS_MODEL_DIR',
    }
    kwargs = {}
    for kwarg, env_name in mapping.items():
        value = _existing_dir_from_env(env_name)
        if value:
            kwargs[kwarg] = value
    return kwargs


class PaddleOcrEngine(OcrEngine):
    name = 'paddleocr'
    def __init__(self):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', **paddleocr_kwargs_from_env())
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
