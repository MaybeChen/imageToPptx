from PIL import Image, ImageStat


def rgb_to_hex(rgb):
    return '#%02X%02X%02X' % tuple(int(max(0, min(255, c))) for c in rgb[:3])


def dominant_hex(image_path, bbox=None, default='#FFFFFF'):
    try:
        im = Image.open(image_path).convert('RGB')
        if bbox:
            x, y, w, h = [int(v) for v in bbox]
            im = im.crop((x, y, x + w, y + h))
        im.thumbnail((32, 32))
        return rgb_to_hex(ImageStat.Stat(im).median)
    except Exception:
        return default


def _luminance(rgb):
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_color_hex(image_path, bbox, default='#1F2937'):
    """Estimate foreground text color from an OCR crop.

    The crop background often dominates, so using a plain median color makes
    native text inherit the card/background color. Estimate the background from
    crop borders, then use high-contrast foreground pixels for the text color.
    """
    try:
        im = Image.open(image_path).convert('RGB')
        x, y, w, h = [int(v) for v in bbox]
        crop = im.crop((x, y, x + max(1, w), y + max(1, h)))
        crop.thumbnail((96, 48))
        width, height = crop.size
        get_pixels = getattr(crop, 'get_flattened_data', crop.getdata)
        pixels = list(get_pixels())
        if not pixels:
            return default

        border = []
        for by in range(height):
            for bx in range(width):
                if bx in (0, width - 1) or by in (0, height - 1):
                    border.append(crop.getpixel((bx, by)))
        background = ImageStat.Stat(crop).median
        if border:
            # ImageStat works on images, so build a compact border strip.
            border_im = Image.new('RGB', (len(border), 1))
            border_im.putdata(border)
            background = ImageStat.Stat(border_im).median
        bg_luma = _luminance(background)
        ranked = sorted(pixels, key=lambda px: abs(_luminance(px) - bg_luma), reverse=True)
        foreground = [px for px in ranked if abs(_luminance(px) - bg_luma) >= 35]
        if not foreground:
            foreground = ranked[:max(1, len(ranked) // 20)]
        fg_im = Image.new('RGB', (len(foreground), 1))
        fg_im.putdata(foreground)
        return rgb_to_hex(ImageStat.Stat(fg_im).median)
    except Exception:
        return default
