from app.utils.files import write_json


def write_quality_report(job_id, manifest, output_path, extra_warnings=None):
    q=manifest.quality
    report={'job_id':job_id,'status':'completed','strategy':manifest.strategy.model_dump(),'ocr':{'engine':q.ocr_engine,'text_count':q.ocr_text_count},'editability':{'text_boxes':q.native_text_count,'shapes':q.shape_count,'lines':sum(e.type in ('line','arrow') for e in manifest.elements),'image_assets':q.image_asset_count,'background_assets':q.background_asset_count},'warnings':list(q.warnings)+list(extra_warnings or []),'limitations':['Full-slide background fallback is disabled by default; enable USE_FULL_SLIDE_BACKGROUND=1 only when visual fidelity is more important than editability.','Complex logos, photos, illustrations, and charts are inserted as image assets in the MVP.','PDF/multi-page input, SAM/SAM3, VLM, inpainting, DrawIO, render diff, and async queues are extension points.']}
    write_json(output_path, report); return report
