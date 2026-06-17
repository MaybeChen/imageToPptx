from __future__ import annotations

import os


def paddleocr_mkldnn_enabled() -> bool:
    """Return whether PaddleOCR oneDNN/MKLDNN acceleration is explicitly enabled."""
    return os.getenv('PADDLEOCR_ENABLE_MKLDNN') == '1'


def configure_paddleocr_runtime() -> None:
    """Apply Paddle CPU runtime defaults before Paddle/PaddleOCR can be imported.

    Paddle reads these flags during import/initialization. Force-disable the
    oneDNN/MKLDNN path by default because it can crash OCR detection with
    `OneDnnContext does not have the input Filter` / `fused_conv2d` errors on
    some CPU Paddle builds. Operators can opt in after validating their runtime
    by setting PADDLEOCR_ENABLE_MKLDNN=1 before the service starts.
    """
    if paddleocr_mkldnn_enabled():
        return
    os.environ['FLAGS_use_mkldnn'] = '0'
    os.environ['FLAGS_use_onednn'] = '0'


configure_paddleocr_runtime()
