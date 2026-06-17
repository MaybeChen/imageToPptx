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

from PIL import Image

from app.pipeline.manifest import build_manifest


def test_build_manifest_preserves_icon_asset_type(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (200, 120), "white").save(source)
    job_root = tmp_path / "job"
    (job_root / "assets").mkdir(parents=True)
    output = job_root / "output"
    output.mkdir()
    job = {
        "source_path": source,
        "job_root": job_root,
        "width_px": 200,
        "height_px": 120,
        "file_name": "source.png",
        "dirs": {"output": output},
    }

    manifest, _ = build_manifest(
        job,
        ocr_items=[],
        segments=[SegmentItem(type="icon", bbox_px=[20, 20, 30, 30], confidence=0.8)],
    )

    icon = next(element for element in manifest.elements if element.type == "icon")
    assert icon.id == "icon_001"
    assert icon.asset_path == "assets/icon_001.png"
    assert manifest.quality.image_asset_count == 1


def test_build_manifest_uses_ocr_bbox_for_smaller_font_and_text_color(tmp_path):
    source = tmp_path / "text_source.png"
    image = Image.new("RGB", (1280, 720), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Hi", fill=(10, 20, 220))
    image.save(source)
    job_root = tmp_path / "text_job"
    (job_root / "assets").mkdir(parents=True)
    output = job_root / "output"
    output.mkdir()
    job = {
        "source_path": source,
        "job_root": job_root,
        "width_px": 1280,
        "height_px": 720,
        "file_name": "text_source.png",
        "dirs": {"output": output},
    }

    manifest, _ = build_manifest(
        job,
        ocr_items=[OcrItem(text="Hi", bbox_px=[18, 18, 40, 20], confidence=0.9)],
        segments=[],
    )

    text = next(element for element in manifest.elements if element.type == "text")
    assert text.style["font_size"] == 10.8
    assert text.style["margin_left"] == 0
    assert text.style["color"].startswith("#")
