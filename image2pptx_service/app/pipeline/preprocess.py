from pathlib import Path
from PIL import Image
from app.config import settings
from app.utils.files import new_job_id, ensure_job_dirs

class UnsupportedImageFormat(ValueError): pass

def prepare_image(upload_path: Path, original_name: str | None = None, job_id: str | None = None):
    suffix = Path(original_name or upload_path.name).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise UnsupportedImageFormat(f'Unsupported image type: {suffix}')
    job_id = job_id or new_job_id()
    job_root = settings.jobs_dir / job_id
    dirs = ensure_job_dirs(job_root)
    source_path = dirs['input'] / 'source.png'
    with Image.open(upload_path) as im:
        rgb = im.convert('RGB')
        rgb.save(source_path, 'PNG')
        width, height = rgb.size
    return {'job_id': job_id, 'job_root': job_root, 'dirs': dirs, 'source_path': source_path, 'width_px': width, 'height_px': height, 'file_name': original_name or upload_path.name}
