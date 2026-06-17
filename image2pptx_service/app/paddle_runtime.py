from __future__ import annotations

import os
import sys
from typing import Any


def paddleocr_mkldnn_enabled() -> bool:
    """Return whether PaddleOCR oneDNN/MKLDNN acceleration is explicitly enabled."""
    return os.getenv('PADDLEOCR_ENABLE_MKLDNN') == '1'


def paddleocr_ir_optim_enabled() -> bool:
    """Return whether PaddleOCR graph/IR optimizations are explicitly enabled."""
    return os.getenv('PADDLEOCR_ENABLE_IR_OPTIM') == '1'


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


def configure_loaded_paddle_runtime() -> None:
    """Apply safe flags to an already-imported Paddle module when available."""
    if paddleocr_mkldnn_enabled():
        return
    paddle: Any | None = sys.modules.get('paddle')
    set_flags = getattr(paddle, 'set_flags', None) if paddle is not None else None
    if set_flags is None:
        return
    for flag_name in ('FLAGS_use_mkldnn', 'FLAGS_use_onednn'):
        try:
            set_flags({flag_name: False})
        except (RuntimeError, ValueError):
            # Paddle versions differ in accepted runtime flags; environment
            # flags and PaddleOCR constructor kwargs still carry the default.
            continue


configure_paddleocr_runtime()
