# image-to-editable-pptx Python Service MVP

A FastAPI service that converts a single image (`png/jpg/jpeg/webp`) into a PowerPoint `.pptx` with a structured `slide_manifest.json`, independent image assets, a preview placeholder, and a quality report.

## Install

```bash
cd image2pptx_service
python -m venv .venv
source .venv/bin/activate
poetry install
```

Optional OCR: install the Tesseract binary for `pytesseract` (or set `TESSERACT_CMD` to its executable path), or run `poetry install --extras paddleocr` to install the PaddleOCR adapter with the CPU `paddlepaddle` runtime. `setuptools` is included because Paddle/PaddleOCR imports still require it in some environments. If OCR is unavailable, the service degrades to the dummy OCR adapter and records a warning.

### Tesseract OCR setup

Install Tesseract OCR on the same machine that runs the service, then restart the service process before submitting image conversion jobs.

Windows common setup:

1. Install Tesseract, for example to `C:\Program Files\Tesseract-OCR\tesseract.exe`.
2. Set `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` for the service process.
3. Restart the service process.
4. Submit the image conversion task again.

Linux/macOS common setup:

1. Install the system `tesseract` package.
2. Confirm the service process can find it with `which tesseract`.
3. If it cannot be found, set `TESSERACT_CMD` to the actual executable path.


## Dependency management

Dependencies are managed with Poetry via `pyproject.toml`. Use `poetry install` for the MVP runtime and test dependencies, or `poetry install --extras paddleocr` when CPU PaddleOCR support is required. The `paddleocr` extra intentionally installs both `paddleocr` and the CPU `paddlepaddle` runtime so the adapter is not left with only the wrapper package. If you need GPU acceleration, install the platform/CUDA-specific `paddlepaddle-gpu` wheel from the official PaddlePaddle index in your deployment image instead of relying on the CPU extra, then install the remaining project dependencies. If `poetry run python -c "import paddle; paddle.utils.run_check()"` reports `ModuleNotFoundError: No module named 'setuptools'`, rerun `poetry install` after pulling this change or run `poetry add setuptools`.



## Edit-Banana-inspired extraction strategy

The service separates OCR text restoration from visual element segmentation instead of asking OCR to classify every object. The local segment adapter uses lightweight prompt-group-style classes that mirror the Edit-Banana pipeline: simple native `shape` regions, small complex `icon` regions, larger `image`/`chart` assets, and `line`/`arrow` connectors. Manifest generation then merges these layers in the order `background -> shape -> image/icon/chart -> line/arrow -> text`, and filters OCR candidates that mostly overlap icon-like visual regions so icon glyphs are less likely to become editable text boxes.

This is still a local OpenCV heuristic adapter, not a SAM/SAM3 implementation. For production-quality extraction comparable to Edit-Banana, replace or extend `app.pipeline.segment.detect_segments` with a model-backed semantic segmenter that emits the same `SegmentItem` categories.

## PaddleOCR offline/corporate-network setup

PaddleOCR downloads detection/recognition/classification model archives on first initialization. In corporate Windows environments this can fail with `SSLCertVerificationError: self signed certificate in certificate chain` while downloading from `paddleocr.bj.bcebos.com`. To avoid runtime downloads, pre-download and extract the PaddleOCR model directories into the fixed project-local directory below:

```text
image2pptx_service/storage/models/paddleocr/ch_PP-OCRv4_det_infer
image2pptx_service/storage/models/paddleocr/ch_PP-OCRv4_rec_infer
image2pptx_service/storage/models/paddleocr/ch_ppocr_mobile_v2.0_cls_infer
```

Then verify from `image2pptx_service` using the project helper. Do not call `PaddleOCR(use_angle_cls=True, lang='ch')` directly unless you also pass model directories, because raw PaddleOCR will still try to download missing models:

```powershell
poetry run python -c "from app.pipeline.ocr import create_paddleocr; create_paddleocr(); print('ok')"
```

The service first uses optional `PADDLEOCR_DET_MODEL_DIR`, `PADDLEOCR_REC_MODEL_DIR`, and `PADDLEOCR_CLS_MODEL_DIR` overrides when present, otherwise it automatically uses the fixed project-local `storage/models/paddleocr` directories. By default the adapter fails fast if the local model set is missing or incomplete, so it will not silently trigger PaddleOCR downloads. Set `PADDLEOCR_ALLOW_DOWNLOAD=1` only if you intentionally want PaddleOCR to download models at startup. The app package applies Paddle runtime flags before PaddleOCR is imported, reapplies safe flags if Paddle has already been loaded, force-disables Paddle oneDNN/MKLDNN, and disables PaddleOCR IR graph optimization by default to avoid CPU `fused_conv2d` runtime errors such as `OneDnnContext does not have the input Filter`; set `PADDLEOCR_ENABLE_MKLDNN=1` or `PADDLEOCR_ENABLE_IR_OPTIM=1` before service startup only if you have verified those acceleration paths work in your runtime. Large model files are ignored by git; keep only `storage/models/.gitkeep` in source control. The `No ccache found` message is only a Paddle warning and is not the reason OCR initialization fails.

## Start service

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
# or
poetry run python run.py
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


## Direct Python usage

For local debugging or tests, call the conversion pipeline directly without starting FastAPI:

```python
from app.service import convert_image_to_pptx

artifacts = convert_image_to_pptx(
    "storage/samples/test.png",
    mode="balanced",
    job_id="local-debug",
)
print(artifacts.pptx_path)
print(artifacts.manifest_path)
print(artifacts.quality_report_path)
```

The helper returns a `ConversionArtifacts` dataclass and uses the same preprocess → OCR → segmentation → manifest → PPTX → preview → quality pipeline as `/convert`.

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

`slide_manifest.json` is the only intermediate representation consumed by the PPTX builder. Coordinates are source-image pixels and every element includes `bbox_px` and `editable`. Complex slides include a `background` element that points to `assets/background.png` as a visual fallback.

## MVP boundaries

- Readable OCR text becomes native PowerPoint text boxes when OCR is available.
- Simple detected rectangles become native shapes.
- The original slide image is inserted as a full-slide visual fallback background, then detected editable elements are layered on top. This improves fidelity for complex business slides but the background itself is not internally editable.
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
