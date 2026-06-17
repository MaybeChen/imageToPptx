from pathlib import Path
import json, uuid

def new_job_id() -> str:
    return uuid.uuid4().hex[:16]

def ensure_job_dirs(root: Path) -> dict[str, Path]:
    paths = {name: root / name for name in ['input','assets','output','debug']}
    for p in paths.values(): p.mkdir(parents=True, exist_ok=True)
    return paths

def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))
