from app.pipeline.ocr import get_ocr_engine
from app.pipeline.segment import detect_segments
from app.pipeline.layout import combine_layout
from app.pipeline.manifest import build_manifest
from app.pipeline.pptx_builder import build_pptx
from app.pipeline.render import render_preview
from app.pipeline.audit import write_quality_report

def run_conversion(job, mode='balanced', ppt_width=13.333, ppt_height=7.5):
    engine, warnings = get_ocr_engine()
    try: ocr_items = engine.detect(str(job['source_path']))
    except Exception as e:
        ocr_items=[]; warnings.append(f'OCR failed with {engine.name}: {e.__class__.__name__}')
    segments = combine_layout(ocr_items, detect_segments(job['source_path'], mode))
    manifest, manifest_path = build_manifest(job, ocr_items, segments, mode, ppt_width, ppt_height, warnings, ocr_engine=engine.name)
    pptx_path = build_pptx(manifest, job['job_root'], job['dirs']['output']/'result.pptx')
    preview_path, preview_warnings = render_preview(job['source_path'], job['dirs']['output']/'preview.png')
    quality = write_quality_report(job['job_id'], manifest, job['dirs']['output']/'quality_report.json', preview_warnings)
    return {'manifest': manifest_path, 'pptx': pptx_path, 'preview': preview_path, 'quality': quality}
