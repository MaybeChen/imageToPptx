from PIL import Image, ImageDraw

from app.utils.colors import text_color_hex


def test_text_color_hex_estimates_foreground_color(tmp_path):
    sample = tmp_path / "text.png"
    image = Image.new("RGB", (220, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Red", fill=(220, 20, 20))
    image.save(sample)

    assert text_color_hex(sample, [15, 15, 90, 40]).startswith("#")
    color = text_color_hex(sample, [15, 15, 90, 40])
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:7], 16)
    assert red > green
    assert red > blue
