from PIL import Image, ImageDraw

from app.service import convert_image_to_pptx


def test_convert_image_to_pptx_direct_python_method(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "jobs_dir", tmp_path / "jobs")
    sample = tmp_path / "direct.png"
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((60, 80, 300, 170), outline="black", fill="#eef2ff")
    draw.text((90, 112), "Direct Call", fill="black")
    draw.rectangle((420, 90, 560, 220), fill="red")
    image.save(sample)

    artifacts = convert_image_to_pptx(sample, mode="balanced", job_id="direct")

    assert artifacts.status == "completed"
    assert artifacts.job_id == "direct"
    assert artifacts.pptx_path.exists()
    assert artifacts.manifest_path.exists()
    assert artifacts.quality_report_path.exists()
    assert artifacts.preview_path.exists()
    assert artifacts.to_dict()["pptx_path"].endswith("result.pptx")
