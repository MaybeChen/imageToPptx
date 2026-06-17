from pathlib import Path
from shutil import copyfile

def render_preview(source_image: Path, output_path: Path) -> tuple[Path, list[str]]:
    copyfile(source_image, output_path)
    return output_path, ['Preview is a reference copy of the normalized source image; install LibreOffice for real PPTX rendering.']
