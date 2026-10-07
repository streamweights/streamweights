"""Draw docs/img/social-preview.png (1280 x 640): the name, the headline, the proof line and
the flow diagram. Needs Pillow (not a dependency of the package).

  python scripts/make_social_preview.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img" / "social-preview.png"
W, H, S = 1280, 640, 2          # drawn at 2x, saved at 1x
INK, MUTE, LINE = "#1f2328", "#59636e", "#8c959f"
CARD, ACCENT, ACCENT_BG = "#f6f8fa", "#0969da", "#ddf4ff"
BIG, BIG_BG = "#bf3989", "#ffeff7"
HELV = "/System/Library/Fonts/Helvetica.ttc"


def font(size, bold=False):
    return ImageFont.truetype(HELV, size * S, index=1 if bold else 0)


img = Image.new("RGB", (W * S, H * S), "white")
d = ImageDraw.Draw(img)


def text(xy, s, size, fill=INK, bold=False, anchor="la"):
    d.text((xy[0] * S, xy[1] * S), s, font=font(size, bold), fill=fill, anchor=anchor)


def box(x0, y0, x1, y1, fill, outline, width=2, r=14):
    d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S, fill=fill,
                        outline=outline, width=width * S)


def dashed_box(x0, y0, x1, y1, fill, outline):
    box(x0, y0, x1, y1, fill, None, 0)
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1),
                             (x1, y0, x1, y1)):
        n = int(max(abs(bx - ax), abs(by - ay)) // 14)
        for i in range(0, n, 2):
            t0, t1 = i / n, min((i + 1) / n, 1)
            d.line([(ax + (bx - ax) * t0) * S, (ay + (by - ay) * t0) * S,
                    (ax + (bx - ax) * t1) * S, (ay + (by - ay) * t1) * S],
                   fill=outline, width=3 * S)


def arrow(x0, y0, x1, y1, color=LINE):
    d.line([x0 * S, y0 * S, x1 * S, y1 * S], fill=color, width=4 * S)
    sign = 1 if x1 > x0 else -1
    d.polygon([(x1 * S, y1 * S), ((x1 - 16 * sign) * S, (y1 - 10) * S),
               ((x1 - 16 * sign) * S, (y1 + 10) * S)], fill=color)


text((64, 40), "streamweights", 40, ACCENT, bold=True)
text((64, 100), "A 70B model doesn't fit on your laptop.", 54, INK, bold=True)
text((64, 166), "Build your own model from it anyway.", 54, INK, bold=True)
text((64, 250), "Measured: Llama 3.3 70B, 141 GB unquantized, run on a 48 GB MacBook Pro.", 27, MUTE)

for i, (name, sub) in enumerate((("evals.jsonl", "the exam"), ("train.jsonl", "your answers"),
                                 ("prompts.jsonl", "questions only"))):
    y = 335 + i * 88
    box(64, y, 324, y + 72, CARD, LINE, 2, 10)
    text((84, y + 12), name, 28, INK, bold=True)
    text((84, y + 44), sub, 20, MUTE)
    arrow(324, y + 36, 428, 430 + (i - 1) * 24)

box(430, 350, 700, 580, ACCENT_BG, ACCENT, 3)
text((565, 440), "spill build", 36, INK, bold=True, anchor="ma")
text((565, 492), "tune, then grade", 22, MUTE, anchor="ma")

dashed_box(820, 335, 1216, 410, BIG_BG, BIG)
text((1018, 352), "70B teacher", 30, INK, bold=True, anchor="ma")
text((1018, 386), "optional", 18, MUTE, anchor="ma")
arrow(820, 372, 704, 400, BIG)

box(820, 450, 1216, 580, ACCENT_BG, ACCENT, 3)
text((1018, 490), "your model", 36, INK, bold=True, anchor="ma")
text((1018, 536), "base + adapter", 22, MUTE, anchor="ma")
arrow(700, 515, 820, 515)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.resize((W, H), Image.LANCZOS).save(OUT, optimize=True)
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
