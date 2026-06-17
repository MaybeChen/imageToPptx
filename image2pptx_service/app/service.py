from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from app.pipeline.preprocess import prepare_image
from app.workers.tasks import run_conversion


@dataclass(frozen=True)
class ConversionArtifacts:
    """File-system artifacts returned by direct Python conversion calls."""

    job_id: str
    status: str
    job_root: Path
    pptx_path: Path
    manifest_path: Path
    quality_report_path: Path
    preview_path: Path

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-friendly representation for tests or scripts."""
        data = asdict(self)
        return {key: str(value) for key, value in data.items()}


def convert_image_to_pptx(
    image_path: str | Path,
    *,
    mode: str = "balanced",
    ppt_width: float = 13.333,
    ppt_height: float = 7.5,
    job_id: str | None = None,
    original_name: str | None = None,
) -> ConversionArtifacts:
    """Convert a local image to PPTX without going through the HTTP API.

    This helper is intentionally thin: it uses the same preprocess and worker
    pipeline as the FastAPI endpoint, making it convenient for local tests,
    notebooks, and future agent/skill wrappers.
    """
    if mode not in {"fast", "balanced", "editable"}:
        raise ValueError("mode must be fast, balanced, or editable")

    input_path = Path(image_path)
    job = prepare_image(
        input_path,
        original_name=original_name or input_path.name,
        job_id=job_id,
    )
    run_conversion(job, mode=mode, ppt_width=ppt_width, ppt_height=ppt_height)
    output_dir = job["dirs"]["output"]
    return ConversionArtifacts(
        job_id=job["job_id"],
        status="completed",
        job_root=job["job_root"],
        pptx_path=output_dir / "result.pptx",
        manifest_path=output_dir / "slide_manifest.json",
        quality_report_path=output_dir / "quality_report.json",
        preview_path=output_dir / "preview.png",
    )
