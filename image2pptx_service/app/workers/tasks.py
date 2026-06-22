from app.pipeline.ocr import DummyOcrEngine, get_ocr_engine
from app.pipeline.segment import detect_segments
from app.pipeline.layout import combine_layout
from app.pipeline.manifest import build_manifest
from app.pipeline.pptx_builder import build_pptx
from app.pipeline.render import render_preview
from app.pipeline.audit import write_quality_report


def _run_ocr_with_fallbacks(image_path: str):
    """Run OCR and retry with cheaper fallbacks if the selected engine fails at detect time."""
    engine, warnings = get_ocr_engine()
    attempted = {engine.name}
    try:
        return engine.detect(image_path), engine.name, warnings
    except Exception as e:
        warnings.append(f'OCR failed with {engine.name}: {e.__class__.__name__}: {e}')

    for prefer in ('tesseract',):
        fallback_engine, fallback_warnings = get_ocr_engine(prefer)
        warnings.extend(fallback_warnings)
        if fallback_engine.name in attempted:
            continue
        attempted.add(fallback_engine.name)
        try:
            return fallback_engine.detect(image_path), fallback_engine.name, warnings
        except Exception as e:
            warnings.append(f'OCR failed with {fallback_engine.name}: {e.__class__.__name__}: {e}')

    warnings.append('Using Dummy OCR engine after all OCR engines failed; no native text boxes may be created.')
    return [], DummyOcrEngine.name, warnings


def run_conversion(job, mode='balanced', ppt_width=13.333, ppt_height=7.5):
    ocr_items, ocr_engine, warnings = _run_ocr_with_fallbacks(str(job['source_path']))
    segments = combine_layout(
        ocr_items,
        detect_segments(job['source_path'], mode, debug_image_path=job['dirs']['output'] / 'yolo_detections.png'),
    )
    manifest, manifest_path = build_manifest(job, ocr_items, segments, mode, ppt_width, ppt_height, warnings, ocr_engine=ocr_engine)
    pptx_path = build_pptx(manifest, job['job_root'], job['dirs']['output']/'result.pptx')
    preview_path, preview_warnings = render_preview(job['source_path'], job['dirs']['output']/'preview.png')
    quality = write_quality_report(job['job_id'], manifest, job['dirs']['output']/'quality_report.json', preview_warnings)
    return {'manifest': manifest_path, 'pptx': pptx_path, 'preview': preview_path, 'quality': quality}
