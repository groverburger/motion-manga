"""Panel detection via image-gen masking.

Hypothesis: GPT Image 2 understands panel layout semantically. If we ask it
to output the SAME page with only panel N visible and everything else solid
black, we can extract the polygon by thresholding the result.

This uses the image model AS the panel detector — outsourcing the hard
semantic question (which pixels belong to panel N) to the most capable
system we have.
"""
import base64, json, pathlib, sys
from openai import OpenAI
import cv2
import numpy as np
from PIL import Image, ImageDraw

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "page1.png")
panel_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 1
stem = src.stem

client = OpenAI(api_key=pathlib.Path.home().joinpath("Documents/openai_api_key.txt").read_text().strip())

# Describe panel 1 concretely so the model knows which one to keep.
# For page1.png: panel 1 is the top "eyes close-up" panel.
# We phrase it positionally rather than content-wise so it generalises.
reading_order_desc = {
    1: "the very first panel — the topmost panel that the reader reads first",
    2: "the second panel in reading order (top-to-bottom, left-to-right)",
    3: "the third panel in reading order",
    4: "the fourth panel in reading order",
    5: "the fifth panel in reading order",
    6: "the sixth panel in reading order",
}

prompt = f"""Produce a BINARY MASK image the same size as the input:

- Fill the area of {reading_order_desc.get(panel_idx, f'panel {panel_idx}')} with PURE WHITE (#FFFFFF).
- Fill EVERYTHING ELSE with PURE BLACK (#000000).

Critical constraints:
- Output ONLY pure white and pure black — no grayscale, no texture, no
  original page content should be visible.
- The boundary between the white region and the black region must follow the
  panel's actual gutter/border precisely — trace its real edges, even if they
  are angled or if a character figure breaks the panel frame.
- Do NOT crop or resize — match the input dimensions exactly.
"""

print(f"Generating panel-{panel_idx}-isolated version of {src}...")
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
out_path = pathlib.Path(f"_{stem}_iso_p{panel_idx}.png")
if b64:
    out_path.write_bytes(base64.b64decode(b64))
else:
    import urllib.request
    urllib.request.urlretrieve(resp.data[0].url, out_path)
print(f"Saved {out_path}")

# Extract the polygon from the isolated-panel image.
iso = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
# Anywhere that's not nearly-black is the panel.
_, panel_mask = cv2.threshold(iso, 30, 255, cv2.THRESH_BINARY)
cv2.imwrite(f"_{stem}_iso_p{panel_idx}_mask.png", panel_mask)

contours, _ = cv2.findContours(panel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("No panel found in masked output!")
    sys.exit(1)
cnt = max(contours, key=cv2.contourArea)
perim = cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, 0.003 * perim, True)
pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
print(f"\nPanel {panel_idx} polygon: {len(pts)} vertices")
for p in pts:
    print(f"  {p}")

# Visualize — overlay the detected polygon on the ORIGINAL page
orig = Image.open(src).convert("RGBA")
overlay = Image.new("RGBA", orig.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)
draw.polygon(pts, fill=(0, 255, 0, 110), outline=(255, 255, 255, 255))
Image.alpha_composite(orig, overlay).convert("RGB").save(f"{stem}_iso_p{panel_idx}_verify.png")
print(f"Saved {stem}_iso_p{panel_idx}_verify.png")
