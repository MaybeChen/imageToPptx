from pathlib import Path
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from app.config import settings
from app.schemas import ConvertResponse
from app.pipeline.preprocess import UnsupportedImageFormat
from app.service import convert_image_to_pptx

app = FastAPI(title='image-to-editable-pptx MVP', version='0.1.0')

@app.get('/health')
def health(): return {'status':'ok'}

@app.post('/convert', response_model=ConvertResponse)
async def convert(file: UploadFile = File(...), mode: str = Form('balanced'), ppt_width: float = Form(13.333), ppt_height: float = Form(7.5)):
    if mode not in {'fast','balanced','editable'}: raise HTTPException(400, 'mode must be fast, balanced, or editable')
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or 'input.png').suffix) as tmp:
        tmp.write(await file.read()); tmp_path=Path(tmp.name)
    try:
        artifacts = convert_image_to_pptx(tmp_path, original_name=file.filename, mode=mode, ppt_width=ppt_width, ppt_height=ppt_height)
    except UnsupportedImageFormat as e:
        raise HTTPException(400, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)
    jid=artifacts.job_id
    return ConvertResponse(job_id=jid,status='completed',pptx_url=f'/download/{jid}/pptx',manifest_url=f'/download/{jid}/manifest',quality_report_url=f'/download/{jid}/quality',preview_url=f'/download/{jid}/preview',yolo_debug_url=f'/download/{jid}/yolo_debug')

DOWNLOADS={'pptx':('result.pptx','application/vnd.openxmlformats-officedocument.presentationml.presentation'),'manifest':('slide_manifest.json','application/json'),'quality':('quality_report.json','application/json'),'preview':('preview.png','image/png'),'yolo_debug':('yolo_detections.png','image/png')}
@app.get('/download/{job_id}/{kind}')
def download(job_id: str, kind: str):
    if kind not in DOWNLOADS: raise HTTPException(404, 'unknown artifact')
    filename, media=DOWNLOADS[kind]; path=settings.jobs_dir/job_id/'output'/filename
    if not path.exists(): raise HTTPException(404, f'{kind} artifact not found')
    return FileResponse(path, media_type=media, filename=filename)
