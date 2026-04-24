"""Test two mask-generation prompts on the same page:
   (a) FILLS: all panels filled solid white, gutters+outside solid black
   (b) EDGES: only the panel border/gutter lines drawn in white, rest black
"""
import base64, pathlib, sys
from openai import OpenAI
import cv2
import numpy as np
from PIL import Image, ImageDraw

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "page1.png")
mode = sys.argv[2] if len(sys.argv) > 2 else "fills"   # "fills" or "edges"
stem = src.stem

client = OpenAI(api_key=pathlib.Path.home().joinpath("Documents/openai_api_key.txt").read_text().strip())

PROMPTS = {
    "fills": """Output a BINARY DIAGRAM the same size as the input image.

- Fill the interior of EVERY comic panel with PURE WHITE (#FFFFFF).
- Fill all gutters, the background, and everything outside any panel with PURE BLACK (#000000).
- Do NOT include any manga artwork, characters, or line art from the original page.
- Do NOT use any grayscale — output must be strictly two colors, white and black.
- The white/black boundary must precisely follow each panel's actual gutter/border edges,
  including angled edges and any places where a character figure breaks the panel frame.
- Output at the same resolution as the input.
""",

    "edges": """Output a BINARY LINE DRAWING the same size as the input image.

- Draw the panel borders / gutter lines in PURE WHITE (#FFFFFF), 4 pixels thick.
- Fill everything else — panel interiors AND outside-panel areas — with PURE BLACK (#000000).
- Do NOT include any manga artwork, characters, or line art from the original page.
- Do NOT use any grayscale — output must be strictly two colors, white and black.
- The white lines must precisely trace each panel's actual gutter/border edges,
  including angled edges and any places where a character figure breaks the panel frame.
- Output at the same resolution as the input.
""",
}

prompt = PROMPTS[mode]
out_path = pathlib.Path(f"_{stem}_mask_{mode}.png")
print(f"Generating {mode} mask for {src}...")
with open(src, "rb") as f:
    resp = client.images.edit(
        model="gpt-image-2",
        image=f,
        prompt=prompt,
        size="1024x1536",
        quality="high",
        n=1,
    )
b64 = resp.data[0].b64_json
if b64:
    out_path.write_bytes(base64.b64decode(b64))
else:
    import urllib.request
    urllib.request.urlretrieve(resp.data[0].url, out_path)
print(f"Saved {out_path}")

# Overlay the result on the original page for comparison.
gen = Image.open(out_path).convert("L").resize((1024, 1536))
orig = Image.open(src).convert("RGBA")
overlay = Image.new("RGBA", orig.size, (0, 0, 0, 0))
for y in range(1536):
    for x in range(1024):
        v = gen.getpixel((x, y))
        if v > 180:
            overlay.putpixel((x, y), (0, 255, 0, 130))
Image.alpha_composite(orig, overlay).convert("RGB").save(f"{stem}_mask_{mode}_overlay.png")
print(f"Saved {stem}_mask_{mode}_overlay.png")
