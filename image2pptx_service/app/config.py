from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    base_dir: Path = Path(__file__).resolve().parents[1]
    storage_dir: Path = base_dir / 'storage'
    jobs_dir: Path = storage_dir / 'jobs'
    allowed_extensions: set[str] = {'.png', '.jpg', '.jpeg', '.webp'}
    default_ppt_width: float = 13.333
    default_ppt_height: float = 7.5

settings = Settings()
settings.jobs_dir.mkdir(parents=True, exist_ok=True)
