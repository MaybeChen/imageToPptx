from pathlib import Path
from shutil import copyfile
import os
from app.schemas import *
from app.utils.colors import dominant_hex, text_color_hex
from app.utils.image import save_crop
from app.utils.files import write_json




def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ('1', 'true', 'yes', 'on')


def _compact_short_text_candidate(item, slide_area) -> bool:
    text = (item.text or '').strip()
    if not text or len(text) > 2:
        return False
    _, _, w, h = item.bbox_px
    if w <= 0 or h <= 0:
        return True
    aspect = w / h
    area_ratio = _area(item.bbox_px) / max(1, slide_area)
    return area_ratio <= _env_float('OCR_SHORT_TEXT_MAX_AREA_RATIO', 0.006) and 0.45 <= aspect <= 2.2

def _area(bbox):
    return max(0, bbox[2]) * max(0, bbox[3])


def _intersection_area(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def _is_icon_like_segment(seg, slide_area):
    if seg.type == 'icon':
        return True
    if seg.type != 'image':
        return False
    _, _, w, h = seg.bbox_px
    if w <= 0 or h <= 0:
        return False
    area_ratio = _area(seg.bbox_px) / max(1, slide_area)
    aspect = w / h
    return area_ratio <= 0.04 and 0.4 <= aspect <= 2.5


def _ocr_item_overlaps_icon(item, icon_segments):
    item_area = _area(item.bbox_px)
    if item_area <= 0:
        return False
    return any(_intersection_area(item.bbox_px, seg.bbox_px) / item_area >= 0.6 for seg in icon_segments)




def _is_ocr_suppression_segment(seg, slide_area) -> bool:
    if seg.type == 'background':
        return False
    if seg.type in ('icon', 'chart', 'table'):
        return True
    if seg.type != 'image':
        return False
    area_ratio = _area(seg.bbox_px) / max(1, slide_area)
    return area_ratio <= _env_float('OCR_VISUAL_ASSET_MAX_AREA_RATIO', 0.12)


def _ocr_item_overlaps_suppression_zone(item, suppression_segments) -> bool:
    item_area = _area(item.bbox_px)
    if item_area <= 0:
        return False
    min_coverage = _env_float('OCR_VISUAL_ASSET_MIN_COVERAGE', 0.6)
    return any(_intersection_area(item.bbox_px, seg.bbox_px) / item_area >= min_coverage for seg in suppression_segments)

def filter_ocr_items_for_manifest(ocr_items, segments, width_px, height_px):
    """Drop OCR boxes that are more likely to be icon glyphs than slide text."""
    slide_area = width_px * height_px
    icon_segments = [seg for seg in segments if _is_icon_like_segment(seg, slide_area)]
    suppression_segments = [seg for seg in segments if _is_ocr_suppression_segment(seg, slide_area)]
    kept = []
    removed = 0
    filter_short_glyphs = _env_flag('OCR_FILTER_SHORT_GLYPHS', True)
    filter_visual_assets = _env_flag('OCR_FILTER_VISUAL_ASSET_TEXT', False)
    for item in ocr_items:
        overlaps_icon = filter_visual_assets and bool(icon_segments) and _ocr_item_overlaps_icon(item, icon_segments)
        overlaps_visual_asset = filter_visual_assets and _ocr_item_overlaps_suppression_zone(item, suppression_segments)
        compact_glyph = filter_short_glyphs and _compact_short_text_candidate(item, slide_area)
        if overlaps_icon or overlaps_visual_asset or compact_glyph:
            removed += 1
        else:
            kept.append(item)
    return kept, removed


def build_manifest(job, ocr_items, segments, mode='balanced', ppt_width=13.333, ppt_height=7.5, warnings=None, ocr_engine='unknown'):
    elements = []
    warnings = list(warnings or [])
    ocr_items, filtered_icon_ocr_count = filter_ocr_items_for_manifest(ocr_items, segments, job['width_px'], job['height_px'])
    if filtered_icon_ocr_count:
        warnings.append(f'Filtered {filtered_icon_ocr_count} OCR text candidate(s) that looked like icon glyphs or overlapped icon/chart/table/image visual asset regions.')

    use_full_slide_background = _env_flag('USE_FULL_SLIDE_BACKGROUND', False)
    background_strategy = 'none'
    if use_full_slide_background:
        background_rel = 'assets/background.png'
        copyfile(job['source_path'], job['job_root'] / background_rel)
        elements.append(ManifestElement(
            id='background_001',
            type='background',
            asset_path=background_rel,
            bbox_px=[0, 0, job['width_px'], job['height_px']],
            editable=False,
            confidence=1.0,
            editable_note='Full-slide source image inserted as optional visual fallback background.',
        ))
        background_strategy = 'image_fallback'
        warnings.append('Full-slide source image is used as an optional visual fallback background; set USE_FULL_SLIDE_BACKGROUND=0 to disable it.')

    for idx, seg in enumerate(segments, 1):
        if seg.type == 'shape':
            elements.append(ManifestElement(id=f'shape_{idx:03d}', type='shape', shape=seg.shape or 'rect', bbox_px=seg.bbox_px, editable=True, confidence=seg.confidence, style={'fill': dominant_hex(job['source_path'], seg.bbox_px), 'stroke':'#D1D5DB','stroke_width':1.0}))
        elif seg.type in ('line','arrow'):
            elements.append(ManifestElement(id=f'line_{idx:03d}', type=seg.type, bbox_px=seg.bbox_px, editable=True, confidence=seg.confidence, style={'stroke':'#111827','stroke_width':1.5}))
        else:
            asset_type = seg.type if seg.type in ('image', 'icon', 'chart', 'table') else 'image'
            asset_rel=f'assets/{asset_type}_{idx:03d}.png'; asset_path=job['job_root']/asset_rel
            save_crop(job['source_path'], seg.bbox_px, asset_path)
            elements.append(ManifestElement(id=f'{asset_type}_{idx:03d}', type=asset_type, asset_path=asset_rel, bbox_px=seg.bbox_px, editable=False, confidence=seg.confidence, editable_note='Inserted as independent image asset to preserve complex visual details.'))
    scale = ppt_height / job['height_px'] * 72
    for idx, item in enumerate(ocr_items, 1):
        fs=max(6, min(54, item.bbox_px[3]*scale*_env_float('OCR_FONT_SCALE', 0.62)))
        elements.append(ManifestElement(id=f'text_{idx:03d}', type='text', text=item.text, bbox_px=item.bbox_px, editable=True, confidence=item.confidence, style={'font_size': round(fs,1), 'font_family':'Microsoft YaHei', 'color': text_color_hex(job['source_path'], item.bbox_px), 'bold': False, 'margin_left': 0, 'margin_right': 0, 'margin_top': 0, 'margin_bottom': 0}))
    elements.sort(key=lambda e: {'background':0,'shape':1,'image':2,'icon':2,'chart':2,'table':2,'line':3,'arrow':3,'text':4}.get(e.type,9))
    quality=ManifestQuality(ocr_text_count=len(ocr_items), native_text_count=sum(e.type=='text' for e in elements), shape_count=sum(e.type=='shape' for e in elements), image_asset_count=sum(e.type in ('image','icon','chart','table') for e in elements), background_asset_count=sum(e.type=='background' for e in elements), ocr_engine=ocr_engine, warnings=warnings)
    editable_native = quality.native_text_count + quality.shape_count + sum(e.type in ('line','arrow') for e in elements)
    edit='medium' if editable_native else 'low'
    manifest=SlideManifest(source=SourceInfo(file_name=job['file_name'], width_px=job['width_px'], height_px=job['height_px']), slide=SlideInfo(width_in=ppt_width, height_in=ppt_height), strategy=StrategyInfo(mode=mode, background=background_strategy, editability_level=edit), elements=elements, quality=quality)
    out=job['dirs']['output']/ 'slide_manifest.json'
    write_json(out, manifest.model_dump(exclude_none=True))
    return manifest, out
