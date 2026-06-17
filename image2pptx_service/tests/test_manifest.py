from app.schemas import SlideManifest, SourceInfo, SlideInfo, StrategyInfo, ManifestElement, ManifestQuality

def test_manifest_schema_requires_core_fields():
    m=SlideManifest(source=SourceInfo(file_name='x.png', width_px=100, height_px=100), slide=SlideInfo(), strategy=StrategyInfo(), elements=[ManifestElement(id='text_001', type='text', text='Hello', bbox_px=[1,2,3,4], editable=True)], quality=ManifestQuality())
    d=m.model_dump()
    assert {'version','source','slide','elements'} <= set(d)
    for e in d['elements']:
        assert {'id','type','bbox_px','editable'} <= set(e)
        assert len(e['bbox_px']) == 4
        assert all(isinstance(n, (int,float)) for n in e['bbox_px'])

from app.pipeline.manifest import filter_ocr_items_for_manifest
from app.schemas import OcrItem, SegmentItem


def test_filter_ocr_items_for_manifest_drops_icon_like_overlap():
    ocr_items = [
        OcrItem(text="A", bbox_px=[102, 102, 18, 18], confidence=0.82),
        OcrItem(text="Revenue", bbox_px=[200, 100, 120, 32], confidence=0.91),
    ]
    segments = [SegmentItem(type="image", bbox_px=[96, 96, 40, 40], confidence=0.6)]

    kept, removed = filter_ocr_items_for_manifest(ocr_items, segments, width_px=1000, height_px=600)

    assert removed == 1
    assert [item.text for item in kept] == ["Revenue"]


def test_filter_ocr_items_for_manifest_keeps_text_over_large_image_region():
    ocr_items = [OcrItem(text="Quarterly Results", bbox_px=[120, 120, 220, 36], confidence=0.93)]
    segments = [SegmentItem(type="image", bbox_px=[80, 80, 640, 360], confidence=0.6)]

    kept, removed = filter_ocr_items_for_manifest(ocr_items, segments, width_px=1000, height_px=600)

    assert removed == 0
    assert kept == ocr_items
