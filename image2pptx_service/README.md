# image-to-editable-pptx Python Service MVP

A FastAPI service that converts a single image (`png/jpg/jpeg/webp`) into a PowerPoint `.pptx` with a structured `slide_manifest.json`, independent image assets, a preview placeholder, and a quality report.

## Install

```bash
cd image2pptx_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional OCR: install the Tesseract binary or add PaddleOCR. If OCR is unavailable, the service degrades to the dummy OCR adapter and records a warning.

## Start service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# or
python run.py
```

## API

- `GET /health` returns `{"status":"ok"}`.
- `POST /convert` accepts multipart form fields:
  - `file`: image file
  - `mode`: `fast`, `balanced`, or `editable` (default `balanced`)
  - `ppt_width`: default `13.333`
  - `ppt_height`: default `7.5`
- Downloads:
  - `/download/{job_id}/pptx`
  - `/download/{job_id}/manifest`
  - `/download/{job_id}/quality`
  - `/download/{job_id}/preview`

## curl example

```bash
curl -X POST "http://localhost:8000/convert" \
  -F "file=@storage/samples/test.png" \
  -F "mode=balanced"
```

## Output files

Each job is written under `storage/jobs/{job_id}/`:

```text
input/source.png
assets/*.png
output/result.pptx
output/slide_manifest.json
output/quality_report.json
output/preview.png
debug/
```

`slide_manifest.json` is the only intermediate representation consumed by the PPTX builder. Coordinates are source-image pixels and every element includes `bbox_px` and `editable`.

## MVP boundaries

- Readable OCR text becomes native PowerPoint text boxes when OCR is available.
- Simple detected rectangles become native shapes.
- Complex visuals, logos, charts, photos, and icons are independent image assets, not internally editable.
- Preview is currently a reference copy of the normalized source image. Install LibreOffice and extend `pipeline/render.py` for true PPTX rendering.
- The service does not use image-generation, inpainting, or sub-agent/page-worker flows.

## Roadmap

- PDF and multi-image multi-page input.
- SAM/SAM3 segmentation adapter.
- VLM semantic classification adapter.
- Image editing/inpainting plugin interface.
- DrawIO XML import/export and DrawIO XML to PPTX conversion.
- True render diff and visual QA.
- Async queue-backed jobs.
- Agent/Skill wrapper that calls this service API.
