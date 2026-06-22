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
            _FakeBox(1, 0.83, [100, 110, 200, 210]),
            _FakeBox(2, 0.72, [20, 200, 160, 206]),
        ]


class _FakeYOLO:
    names = _FakeResult.names

    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, *args, **kwargs):
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

    segments = detect_segments_with_yolo(image)

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

    message = _format_yolo_exception(exc)

    assert "Windows PyTorch DLL dependency load failed" in message
    assert "Microsoft Visual C++ Redistributable 2015-2022" in message
    assert "poetry run python -m pip install --force-reinstall torch torchvision" in message
