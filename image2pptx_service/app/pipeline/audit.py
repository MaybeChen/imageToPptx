from app.utils.files import write_json

def write_quality_report(job_id, manifest, output_path, extra_warnings=None):
    q=manifest.quality
    report={'job_id':job_id,'status':'completed','editability':{'text_boxes':q.native_text_count,'shapes':q.shape_count,'lines':sum(e.type in ('line','arrow') for e in manifest.elements),'image_assets':q.image_asset_count},'warnings':list(q.warnings)+list(extra_warnings or []),'limitations':['Complex logos, photos, illustrations, and charts are inserted as image assets in the MVP.','PDF/multi-page input, SAM/SAM3, VLM, inpainting, DrawIO, render diff, and async queues are extension points.']}
    write_json(output_path, report); return report
