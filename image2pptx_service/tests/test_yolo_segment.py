import sys
import types

import pytest

from app.pipeline.segment import (
    detect_segments_with_yolo,
    find_yolo_model_path,
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
