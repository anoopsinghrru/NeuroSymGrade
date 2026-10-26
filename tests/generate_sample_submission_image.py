"""
Generates a synthetic handwriting-style submission image (Scenario 2 text) for
OCR smoke-testing and demos, so the project doesn't need to ship a scanned photo.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SAMPLE_TEXT = [
    "Step 1: Velocity is v = u + a*t = 0 + 10*2 = 25",
    "Step 2: Displacement is s = (u + v)/2 * t = (0 + 25)/2 * 2 = 25",
    "Step 3: Work done is W = F * s = 20 * 25 = 500",
]

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoepr.ttf",
    r"C:\Windows\Fonts\LHANDW.TTF",
    r"C:\Windows\Fonts\comic.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def generate(output_path: str) -> str:
    scale = 3
    font = _load_font(32 * scale)
    width, height = 1300 * scale, 260 * scale
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    y = 30 * scale
    for line in SAMPLE_TEXT:
        draw.text((30 * scale, y), line, fill=0, font=font)
        y += 70 * scale
    img.save(output_path)
    return output_path


if __name__ == "__main__":
    out = generate(str(Path(__file__).parent.parent / "ui" / "static" / "sample_handwritten_submission.png"))
    print(f"Sample submission image written to: {out}")
