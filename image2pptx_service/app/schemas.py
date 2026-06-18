from __future__ import annotations
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, field_validator

BBox = list[float]

class OcrItem(BaseModel):
    text: str
    bbox_px: BBox
    confidence: float = 0.0

class SegmentItem(BaseModel):
    type: Literal['shape','image','line','arrow','background','icon','chart','table']
    bbox_px: BBox
    confidence: float = 0.0
    shape: Optional[str] = None
    style: dict[str, Any] = Field(default_factory=dict)
    asset_path: Optional[str] = None

class ManifestElement(BaseModel):
    id: str
    type: str
    bbox_px: BBox
    editable: bool
    confidence: float = 0.0
    text: Optional[str] = None
    shape: Optional[str] = None
    asset_path: Optional[str] = None
    style: dict[str, Any] = Field(default_factory=dict)
    editable_note: Optional[str] = None

    @field_validator('bbox_px')
    @classmethod
    def bbox_has_four_numbers(cls, v):
        if len(v) != 4 or not all(isinstance(n, (int, float)) for n in v):
            raise ValueError('bbox_px must contain four numbers')
        return v

class SourceInfo(BaseModel):
    file_name: str
    width_px: int
    height_px: int

class SlideInfo(BaseModel):
    width_in: float = 13.333
    height_in: float = 7.5

class StrategyInfo(BaseModel):
    mode: Literal['fast','balanced','editable'] = 'balanced'
    background: str = 'image_fallback'
    editability_level: Literal['low','medium','high'] = 'medium'

class ManifestQuality(BaseModel):
    ocr_text_count: int = 0
    native_text_count: int = 0
    shape_count: int = 0
    image_asset_count: int = 0
    background_asset_count: int = 0
    ocr_engine: str = 'unknown'
    warnings: list[str] = Field(default_factory=list)

class SlideManifest(BaseModel):
    version: str = '0.1.0'
    source: SourceInfo
    slide: SlideInfo
    strategy: StrategyInfo
    elements: list[ManifestElement]
    quality: ManifestQuality

class ConvertResponse(BaseModel):
    job_id: str
    status: str
    pptx_url: str
    manifest_url: str
    quality_report_url: str
    preview_url: str
