import os
import sys

import pytest

from app.pipeline.ocr import (
    paddleocr_kwargs_from_env,
    paddleocr_kwargs_from_local_models,
    configure_loaded_paddle_runtime,
    configure_paddleocr_runtime,
    get_ocr_engine,
    paddleocr_model_kwargs,
    paddleocr_runtime_kwargs,
)


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


def test_paddleocr_kwargs_from_project_local_models(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.delenv("PADDLEOCR_DET_MODEL_DIR", raising=False)
    monkeypatch.delenv("PADDLEOCR_REC_MODEL_DIR", raising=False)
    monkeypatch.delenv("PADDLEOCR_CLS_MODEL_DIR", raising=False)
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    base = settings.storage_dir / "models" / "paddleocr"
    det = base / "ch_PP-OCRv4_det_infer"
    rec = base / "ch_PP-OCRv4_rec_infer"
    cls = base / "ch_ppocr_mobile_v2.0_cls_infer"
    for path in (det, rec, cls):
        path.mkdir(parents=True)

    assert paddleocr_kwargs_from_local_models() == {
        "det_model_dir": str(det),
        "rec_model_dir": str(rec),
        "cls_model_dir": str(cls),
    }
    assert paddleocr_model_kwargs() == {
        "det_model_dir": str(det),
        "rec_model_dir": str(rec),
        "cls_model_dir": str(cls),
    }


def test_paddleocr_kwargs_from_project_local_models_rejects_missing_set(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.delenv("PADDLEOCR_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")

    with pytest.raises(FileNotFoundError):
        paddleocr_model_kwargs()


def test_paddleocr_kwargs_allows_download_when_explicitly_enabled(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.setenv("PADDLEOCR_ALLOW_DOWNLOAD", "1")
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")

    assert paddleocr_model_kwargs() == {}


def test_paddleocr_kwargs_from_project_local_models_rejects_partial_set(monkeypatch, tmp_path):
    from app.config import settings

    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    (settings.storage_dir / "models" / "paddleocr" / "ch_PP-OCRv4_det_infer").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        paddleocr_kwargs_from_local_models()


def test_paddleocr_runtime_defaults_disable_mkldnn(monkeypatch):
    monkeypatch.delenv("PADDLEOCR_ENABLE_MKLDNN", raising=False)
    monkeypatch.setenv("FLAGS_use_mkldnn", "1")
    monkeypatch.setenv("FLAGS_use_onednn", "1")

    configure_paddleocr_runtime()

    assert paddleocr_runtime_kwargs() == {"enable_mkldnn": False, "ir_optim": False}
    assert os.environ["FLAGS_use_mkldnn"] == "0"
    assert os.environ["FLAGS_use_onednn"] == "0"


def test_paddleocr_runtime_allows_explicit_mkldnn(monkeypatch):
    monkeypatch.setenv("PADDLEOCR_ENABLE_MKLDNN", "1")
    monkeypatch.delenv("FLAGS_use_mkldnn", raising=False)
    monkeypatch.delenv("FLAGS_use_onednn", raising=False)

    configure_paddleocr_runtime()

    assert paddleocr_runtime_kwargs() == {"enable_mkldnn": True, "ir_optim": False}
    assert "FLAGS_use_mkldnn" not in os.environ
    assert "FLAGS_use_onednn" not in os.environ


def test_tesseract_engine_reports_missing_binary_during_selection(monkeypatch):
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr("app.pipeline.ocr.shutil.which", lambda command: None)

    engine, warnings = get_ocr_engine("tesseract")

    assert engine.name == "dummy"
    assert warnings[0].startswith("Tesseract unavailable: FileNotFoundError")
    assert warnings[-1] == "Using Dummy OCR engine; no native text boxes may be created."


def test_paddleocr_runtime_allows_explicit_ir_optim(monkeypatch):
    monkeypatch.setenv("PADDLEOCR_ENABLE_IR_OPTIM", "1")

    assert paddleocr_runtime_kwargs() == {"enable_mkldnn": False, "ir_optim": True}


def test_configure_loaded_paddle_runtime_disables_supported_flags(monkeypatch):
    calls = []

    class FakePaddle:
        @staticmethod
        def set_flags(flags):
            calls.append(flags)

    monkeypatch.delenv("PADDLEOCR_ENABLE_MKLDNN", raising=False)
    monkeypatch.setitem(sys.modules, "paddle", FakePaddle())

    configure_loaded_paddle_runtime()

    assert {"FLAGS_use_mkldnn": False} in calls
    assert {"FLAGS_use_onednn": False} in calls
