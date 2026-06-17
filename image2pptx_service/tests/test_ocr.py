from pathlib import Path

import pytest

from app.pipeline.ocr import paddleocr_kwargs_from_env


def test_paddleocr_kwargs_from_env(monkeypatch, tmp_path):
    det = tmp_path / "det"
    rec = tmp_path / "rec"
    cls = tmp_path / "cls"
    for path in (det, rec, cls):
        path.mkdir()
    monkeypatch.setenv("PADDLEOCR_DET_MODEL_DIR", str(det))
    monkeypatch.setenv("PADDLEOCR_REC_MODEL_DIR", str(rec))
    monkeypatch.setenv("PADDLEOCR_CLS_MODEL_DIR", str(cls))

    assert paddleocr_kwargs_from_env() == {
        "det_model_dir": str(det),
        "rec_model_dir": str(rec),
        "cls_model_dir": str(cls),
    }


def test_paddleocr_kwargs_from_env_rejects_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("PADDLEOCR_DET_MODEL_DIR", str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError):
        paddleocr_kwargs_from_env()
