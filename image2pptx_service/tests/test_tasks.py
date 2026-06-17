from app.pipeline.ocr import OcrEngine
from app.schemas import OcrItem
from app.workers.tasks import _run_ocr_with_fallbacks


class FailingEngine(OcrEngine):
    name = "paddleocr"

    def detect(self, image_path: str) -> list[OcrItem]:
        raise RuntimeError("paddle runtime broke")


class WorkingEngine(OcrEngine):
    name = "tesseract"

    def detect(self, image_path: str) -> list[OcrItem]:
        return [OcrItem(text="hello", bbox_px=[1, 2, 3, 4], confidence=0.9)]


def test_run_ocr_with_fallbacks_uses_tesseract_when_paddle_detect_fails(monkeypatch):
    def fake_get_ocr_engine(prefer="auto"):
        if prefer == "auto":
            return FailingEngine(), []
        if prefer == "tesseract":
            return WorkingEngine(), []
        raise AssertionError(f"unexpected OCR preference: {prefer}")

    monkeypatch.setattr("app.workers.tasks.get_ocr_engine", fake_get_ocr_engine)

    ocr_items, engine_name, warnings = _run_ocr_with_fallbacks("slide.png")

    assert engine_name == "tesseract"
    assert [item.text for item in ocr_items] == ["hello"]
    assert warnings == ["OCR failed with paddleocr: RuntimeError: paddle runtime broke"]
