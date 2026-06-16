from PIL import Image, ImageStat

def rgb_to_hex(rgb):
    return '#%02X%02X%02X' % tuple(int(max(0,min(255,c))) for c in rgb[:3])

def dominant_hex(image_path, bbox=None, default='#FFFFFF'):
    try:
        im = Image.open(image_path).convert('RGB')
        if bbox:
            x,y,w,h = [int(v) for v in bbox]
            im = im.crop((x,y,x+w,y+h))
        im.thumbnail((32,32))
        return rgb_to_hex(ImageStat.Stat(im).median)
    except Exception:
        return default
