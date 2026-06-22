from pathlib import Path
import sys
import types

import pytest

from app.pipeline.segment import (
    detect_segments_with_yolo,
    find_yolo_model_path,
    _format_yolo_exception,
    _label_to_segment_type,
)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = _Scalar(cls_id)
        self.conf = _Scalar(conf)
        self.xyxy = [xyxy]


class _FakeResult:
    names = {0: "logo", 1: "rectangle", 2: "connector"}

    def __init__(self):
        self.boxes = [
            _FakeBox(0, 0.91, [10, 20, 40, 60]),
            _FakeBox(5, 0.83, [100, 110, 200, 210]),
            _FakeBox(10, 0.72, [20, 200, 160, 206]),
        ]


class _FakeYOLO:
    names = _FakeResult.names
    last_predict_kwargs = None

    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, *args, **kwargs):
        type(self).last_predict_kwargs = kwargs
        return [_FakeResult()]


def test_find_yolo_model_path_uses_env(monkeypatch, tmp_path):
    model = tmp_path / "model.pt"
    model.write_text("fake")
    monkeypatch.setenv("YOLO_MODEL_PATH", str(model))

    assert find_yolo_model_path() == model


def test_find_yolo_model_path_rejects_missing_env(monkeypatch, tmp_path):
    monkeypatch.setenv("YOLO_MODEL_PATH", str(tmp_path / "missing.pt"))

    with pytest.raises(FileNotFoundError):
        find_yolo_model_path()


def test_label_to_segment_type_maps_common_aliases():
    assert _label_to_segment_type("0") == "icon"
    assert _label_to_segment_type("1") == "icon"
    assert _label_to_segment_type("2") == "image"
    assert _label_to_segment_type("3") == "chart"
    assert _label_to_segment_type("4") == "table"
    assert _label_to_segment_type("5") == "shape"
    assert _label_to_segment_type("6") == "shape"
    assert _label_to_segment_type("7") == "shape"
    assert _label_to_segment_type("8") == "shape"
    assert _label_to_segment_type("9") == "shape"
    assert _label_to_segment_type("10") == "line"
    assert _label_to_segment_type("11") == "arrow"
    assert _label_to_segment_type("12") == "background"
    assert _label_to_segment_type("13") == "background"
    assert _label_to_segment_type("14") == "background"
    assert _label_to_segment_type("logo") == "icon"
    assert _label_to_segment_type("picture") == "image"
    assert _label_to_segment_type("connector") == "line"
    assert _label_to_segment_type("unknown") == "image"


def test_detect_segments_with_yolo_maps_boxes_to_segment_items(monkeypatch, tmp_path):
    model = tmp_path / "model.pt"
    model.write_text("fake")
    image = tmp_path / "image.png"
    image.write_text("fake")
    monkeypatch.setenv("YOLO_MODEL_PATH", str(model))
    fake_module = types.SimpleNamespace(YOLO=_FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)

    monkeypatch.delenv("YOLO_CONF", raising=False)

    segments = detect_segments_with_yolo(image)

    assert _FakeYOLO.last_predict_kwargs["conf"] == 0.01
    assert [segment.type for segment in segments] == ["shape", "icon", "line"]
    assert segments[0].shape == "rect"
    assert segments[1].bbox_px == [10.0, 20.0, 30.0, 40.0]


def test_detect_segments_accepts_yolo26_engine(monkeypatch, tmp_path):
    from app.pipeline.segment import detect_segments

    image = tmp_path / "image.png"
    image.write_text("fake")
    monkeypatch.setenv("SEGMENT_ENGINE", "yolo26")
    monkeypatch.setattr("app.pipeline.segment.detect_segments_with_yolo", lambda path: [
        __import__("app.schemas", fromlist=["SegmentItem"]).SegmentItem(type="icon", bbox_px=[1, 2, 3, 4], confidence=0.9)
    ])

    assert detect_segments(image)[0].type == "icon"


def test_label_to_segment_type_ignores_text_region_and_maps_table():
    assert _label_to_segment_type("text_region") is None
    assert _label_to_segment_type("text") is None
    assert _label_to_segment_type("table") == "table"
    assert _label_to_segment_type("4") == "table"


def test_merge_segments_by_layer_keeps_foreground_inside_background():
    from app.pipeline.segment import merge_segments_by_layer
    from app.schemas import SegmentItem

    items = [
        SegmentItem(type="background", bbox_px=[0, 0, 1000, 600], confidence=0.9),
        SegmentItem(type="icon", bbox_px=[100, 100, 40, 40], confidence=0.8),
        SegmentItem(type="shape", shape="rect", bbox_px=[200, 100, 200, 80], confidence=0.8),
    ]

    merged = merge_segments_by_layer(items)

    assert [item.type for item in merged] == ["background", "shape", "icon"]


def test_find_yolo_model_path_prefers_best_pt_in_supported_dirs(monkeypatch, tmp_path):
    from app.pipeline import segment

    first_dir = tmp_path / "storage" / "models" / "yolo"
    second_dir = tmp_path / "models" / "yolo"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    generic = first_dir / "zzz.pt"
    best = second_dir / "best.pt"
    generic.write_text("fake")
    best.write_text("fake")
    monkeypatch.delenv("YOLO_MODEL_PATH", raising=False)
    monkeypatch.setattr(segment, "yolo_model_dirs", lambda: [first_dir, second_dir])

    assert find_yolo_model_path() == best


def test_detect_segments_auto_uses_yolo_when_model_exists(monkeypatch, tmp_path, capsys):
    from app.pipeline import segment
    from app.schemas import SegmentItem

    image = tmp_path / "image.png"
    image.write_text("fake")
    monkeypatch.delenv("SEGMENT_ENGINE", raising=False)
    monkeypatch.setattr(segment, "has_yolo_model", lambda: True)
    monkeypatch.setattr(segment, "detect_segments_with_yolo", lambda path: [
        SegmentItem(type="chart", bbox_px=[10, 20, 30, 40], confidence=0.95)
    ])
    monkeypatch.setattr(segment, "detect_segments_with_opencv", lambda path: [
        SegmentItem(type="image", bbox_px=[1, 2, 3, 4], confidence=0.35)
    ])

    assert segment.detect_segments(image)[0].type == "chart"
    output = capsys.readouterr().out
    assert "Segmentation selected: auto detected YOLO model" in output
    assert "Segmentation result: using YOLO items=1" in output


def test_detect_segments_auto_falls_back_to_opencv_when_yolo_fails(monkeypatch, tmp_path, capsys):
    from app.pipeline import segment
    from app.schemas import SegmentItem

    image = tmp_path / "image.png"
    image.write_text("fake")
    monkeypatch.delenv("SEGMENT_ENGINE", raising=False)
    monkeypatch.setattr(segment, "has_yolo_model", lambda: True)
    monkeypatch.setattr(segment, "detect_segments_with_yolo", lambda path: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(segment, "detect_segments_with_opencv", lambda path: [
        SegmentItem(type="image", bbox_px=[1, 2, 3, 4], confidence=0.35)
    ])

    assert segment.detect_segments(image)[0].type == "image"
    output = capsys.readouterr().out
    assert "Segmentation fallback: YOLO failed with RuntimeError: boom; using OpenCV" in output
    assert "Segmentation result: using OpenCV items=1" in output


def test_format_yolo_exception_explains_windows_torch_dll_error(monkeypatch):
    from app.pipeline import segment

    monkeypatch.setattr(segment.os, "name", "nt")
    exc = OSError(
        '[WinError 127] 找不到指定的程序。 Error loading "D:\\venv\\Lib\\site-packages\\torch\\lib\\shm.dll" or one of its dependencies.'
    )

    monkeypatch.setattr(
        segment,
        "_diagnose_windows_torch_dll_error",
        lambda error: "Missing direct DLL dependencies for shm.dll: c10.dll",
    )

    message = _format_yolo_exception(exc)

    assert "Windows PyTorch DLL dependency load failed" in message
    assert "Diagnostic: Missing direct DLL dependencies for shm.dll: c10.dll" in message
    assert "dependent DLLs is missing or ABI-incompatible" in message
    assert "Microsoft Visual C++ Redistributable 2015-2022" in message
    assert "set YOLO_PYTHON to that python.exe" in message
    assert "poetry run python -m pip install --force-reinstall torch torchvision" in message


def test_diagnose_windows_torch_dll_error_reports_missing_target():
    from app.pipeline import segment

    exc = OSError('Error loading "D:\\venv\\Lib\\site-packages\\torch\\lib\\shm.dll" or one of its dependencies.')

    assert "Loader target does not exist" in segment._diagnose_windows_torch_dll_error(exc)


def test_detect_segments_with_yolo_uses_external_python_when_configured(monkeypatch, tmp_path, capsys):
    import json
    import subprocess

    model = tmp_path / "best.pt"
    image = tmp_path / "image.png"
    model.write_text("fake")
    image.write_text("fake")
    monkeypatch.setenv("YOLO_MODEL_PATH", str(model))
    monkeypatch.setenv("YOLO_PYTHON", "C:/working-yolo/python.exe")

    def fake_run(args, **kwargs):
        output_path = args[5]
        Path(output_path).write_text(json.dumps([
            {"label": "logo", "confidence": 0.9, "xyxy": [1, 2, 11, 22]}
        ]), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("app.pipeline.segment.subprocess.run", fake_run)

    segments = detect_segments_with_yolo(image)

    assert segments[0].type == "icon"
    assert segments[0].bbox_px == [1.0, 2.0, 10.0, 20.0]
    output = capsys.readouterr().out
    assert "YOLO subprocess start: python=C:/working-yolo/python.exe" in output
    assert "YOLO subprocess done: raw_items=1 merged_items=1" in output


def test_write_segment_debug_overlay_creates_annotated_image(tmp_path):
    from PIL import Image
    from app.pipeline.segment import write_segment_debug_overlay
    from app.schemas import SegmentItem

    source = tmp_path / "source.png"
    output = tmp_path / "yolo_detections.png"
    Image.new("RGB", (120, 80), "white").save(source)

    write_segment_debug_overlay(
        source,
        [SegmentItem(type="chart", bbox_px=[10, 10, 50, 30], confidence=0.87)],
        output,
    )

    assert output.exists()
