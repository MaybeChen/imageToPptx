from app.pipeline.segment import merge_segments_by_layer
from app.schemas import SegmentItem


def test_merge_segments_by_layer_prefers_shape_over_duplicate_image():
    items = [
        SegmentItem(type="image", bbox_px=[10, 10, 100, 60], confidence=0.9),
        SegmentItem(type="shape", shape="rect", bbox_px=[12, 12, 96, 56], confidence=0.7),
    ]

    merged = merge_segments_by_layer(items)

    assert len(merged) == 1
    assert merged[0].type == "shape"


def test_merge_segments_by_layer_keeps_distinct_prompt_groups():
    items = [
        SegmentItem(type="icon", bbox_px=[10, 10, 32, 32], confidence=0.62),
        SegmentItem(type="shape", shape="rect", bbox_px=[100, 20, 120, 60], confidence=0.72),
        SegmentItem(type="line", bbox_px=[40, 160, 180, 4], confidence=0.62),
    ]

    merged = merge_segments_by_layer(items)

    assert [item.type for item in merged] == ["shape", "icon", "line"]
