from pathlib import Path
from shutil import copyfile
from app.schemas import *
from app.utils.colors import dominant_hex
from app.utils.image import save_crop
from app.utils.files import write_json


def build_manifest(job, ocr_items, segments, mode='balanced', ppt_width=13.333, ppt_height=7.5, warnings=None, ocr_engine='unknown'):
    elements = []
    warnings = list(warnings or [])

    background_rel = 'assets/background.png'
    copyfile(job['source_path'], job['job_root'] / background_rel)
    elements.append(ManifestElement(
        id='background_001',
        type='background',
        asset_path=background_rel,
        bbox_px=[0, 0, job['width_px'], job['height_px']],
        editable=False,
        confidence=1.0,
        editable_note='Full-slide source image inserted as visual fallback background.',
    ))
    warnings.append('Full-slide source image is used as a visual fallback background; editability is provided by overlaid native elements where detected.')

    for idx, seg in enumerate(segments, 1):
        if seg.type == 'shape':
            elements.append(ManifestElement(id=f'shape_{idx:03d}', type='shape', shape=seg.shape or 'rect', bbox_px=seg.bbox_px, editable=True, confidence=seg.confidence, style={'fill': dominant_hex(job['source_path'], seg.bbox_px), 'stroke':'#D1D5DB','stroke_width':1.0}))
        elif seg.type in ('line','arrow'):
            elements.append(ManifestElement(id=f'line_{idx:03d}', type=seg.type, bbox_px=seg.bbox_px, editable=True, confidence=seg.confidence, style={'stroke':'#111827','stroke_width':1.5}))
        else:
            asset_rel=f'assets/image_{idx:03d}.png'; asset_path=job['job_root']/asset_rel
            save_crop(job['source_path'], seg.bbox_px, asset_path)
            elements.append(ManifestElement(id=f'image_{idx:03d}', type='image', asset_path=asset_rel, bbox_px=seg.bbox_px, editable=False, confidence=seg.confidence, editable_note='Inserted as independent image asset to preserve complex visual details.'))
    scale = ppt_height / job['height_px'] * 72
    for idx, item in enumerate(ocr_items, 1):
        fs=max(8, min(60, item.bbox_px[3]*scale*0.9))
        elements.append(ManifestElement(id=f'text_{idx:03d}', type='text', text=item.text, bbox_px=item.bbox_px, editable=True, confidence=item.confidence, style={'font_size': round(fs,1), 'font_family':'Microsoft YaHei', 'color':'#1F2937', 'bold': False}))
    elements.sort(key=lambda e: {'background':0,'shape':1,'image':2,'line':3,'arrow':3,'text':4}.get(e.type,9))
    quality=ManifestQuality(ocr_text_count=len(ocr_items), native_text_count=sum(e.type=='text' for e in elements), shape_count=sum(e.type=='shape' for e in elements), image_asset_count=sum(e.type=='image' for e in elements), background_asset_count=sum(e.type=='background' for e in elements), ocr_engine=ocr_engine, warnings=warnings)
    editable_native = quality.native_text_count + quality.shape_count + sum(e.type in ('line','arrow') for e in elements)
    edit='medium' if editable_native else 'low'
    manifest=SlideManifest(source=SourceInfo(file_name=job['file_name'], width_px=job['width_px'], height_px=job['height_px']), slide=SlideInfo(width_in=ppt_width, height_in=ppt_height), strategy=StrategyInfo(mode=mode, background='image_fallback', editability_level=edit), elements=elements, quality=quality)
    out=job['dirs']['output']/ 'slide_manifest.json'
    write_json(out, manifest.model_dump(exclude_none=True))
    return manifest, out
