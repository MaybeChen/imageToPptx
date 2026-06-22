import json
from pathlib import Path
from PIL import Image, ImageDraw
from app.pipeline.preprocess import prepare_image
from app.workers.tasks import run_conversion

def test_pipeline_smoke(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, 'jobs_dir', tmp_path/'jobs')
    sample=tmp_path/'sample.png'; im=Image.new('RGB',(640,360),'white'); d=ImageDraw.Draw(im); d.rounded_rectangle((50,80,300,170), outline='black', fill='#eeeeff'); d.text((80,110),'Hello MVP', fill='black'); d.rectangle((420,90,560,220), fill='red'); im.save(sample)
    job=prepare_image(sample, 'sample.png', job_id='smoke')
    run_conversion(job, 'balanced')
    out=job['dirs']['output']
    assert (out/'result.pptx').exists()
    assert (out/'slide_manifest.json').exists()
    assert (out/'quality_report.json').exists()
    manifest = json.loads((out/'slide_manifest.json').read_text(encoding='utf-8'))
    assert manifest['strategy']['background'] == 'none'
    assert manifest['quality']['background_asset_count'] == 0
