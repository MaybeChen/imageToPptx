from pathlib import Path
from PIL import Image

def save_crop(source: Path, bbox, dest: Path) -> None:
    im = Image.open(source).convert('RGB')
    x,y,w,h = [int(max(0,v)) for v in bbox]
    im.crop((x,y,x+w,y+h)).save(dest)
