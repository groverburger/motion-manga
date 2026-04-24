"""Final pipeline: ask GPT Image 2 to render a binary panel-fills mask for the
page, then extract each panel as a polygon contour from the mask.
"""
import base64, json, pathlib, sys
from openai import OpenAI
import cv2
import numpy as np
from PIL import Image, ImageDraw

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "page1.png")
stem = src.stem
H_target, W_target = 1536, 1024

client = OpenAI(api_key=pathlib.Path.home().joinpath("Documents/openai_api_key.txt").read_text().strip())

prompt = """Output a BINARY DIAGRAM the same size as the input image.

Task: fill EVERY comic panel's interior with PURE WHITE (#FFFFFF), and fill
ONLY the gutters between panels with PURE BLACK (#000000).

Count every panel, including:
- Panels that bleed to the edge of the page (they still count as panels — fill
  them white all the way to the page boundary).
- Dark-toned panels (night scenes, black backgrounds) — they are still panels,
  just dark ones. Fill them solid white in the output.
- Small panels, reaction panels, splash panels — all panels.

A panel is any region separated from adjacent panels by a gutter (a thin strip
between panels). Start by counting the panels in the input and make sure every
one of them appears as a white region in your output.

Output constraints:
- Strictly two colors: pure white (#FFFFFF) for panel interiors, pure black
  (#000000) everywhere else. No grayscale. No manga artwork. No characters.
- The white/black boundary must trace each panel's actual gutter edges,
  including angled edges and any notches where a character figure breaks the
  panel frame.
- Same resolution as the input.
"""

mask_path = pathlib.Path(f"_{stem}_mask_fills.png")
if not mask_path.exists():
    print(f"Generating mask for {src}...")
    with open(src, "rb") as f:
        resp = client.images.edit(
            model="gpt-image-2",
            image=f, prompt=prompt,
            size="1024x1536", quality="high", n=1,
        )
    b64 = resp.data[0].b64_json
    if b64:
        mask_path.write_bytes(base64.b64decode(b64))
    else:
        import urllib.request
        urllib.request.urlretrieve(resp.data[0].url, mask_path)
    print(f"Saved {mask_path}")
else:
    print(f"Using cached {mask_path}")

# Load and normalize the mask to the target resolution
m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
if m.shape != (H_target, W_target):
    m = cv2.resize(m, (W_target, H_target), interpolation=cv2.INTER_NEAREST)

# Clean up: Otsu threshold then small morph ops to remove speckle
_, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Small close to heal any 1-px gaps along gutters; small open to drop specks
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

# Find each white region as a separate panel
contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
panels_raw = []
W, H = W_target, H_target
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < (W * H) * 0.01:      # ignore tiny specks
        continue
    if area > (W * H) * 0.9:       # ignore the whole-page frame
        continue
    perim = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.003 * perim, True)
    pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
    panels_raw.append((area, pts))

# Reading order: top-to-bottom, left-to-right (group into rows by median height)
panels_raw.sort(key=lambda p: (-p[0]))
heights = [max(py for _, py in pts) - min(py for _, py in pts) for _, pts in panels_raw]
row_gap = (sorted(heights)[len(heights)//2]) // 2 if heights else 150

def centroid(pts):
    return sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)

with_centroids = [(centroid(p), p) for _, p in panels_raw]
with_centroids.sort(key=lambda r: r[0][1])
rows = []
for (cx, cy), pts in with_centroids:
    if rows and abs(cy - rows[-1][0][0][1]) < row_gap:
        rows[-1].append(((cx, cy), pts))
    else:
        rows.append([((cx, cy), pts)])
for row in rows:
    row.sort(key=lambda r: r[0][0])

panels = {}
i = 1
for row in rows:
    for _, pts in row:
        panels[f"p{i}"] = pts
        i += 1

print(f"\n{len(panels)} panels detected")
for name, pts in panels.items():
    print(f"  {name}: {len(pts)} verts")

# Visualize
viz = Image.open(src).convert("RGBA")
overlay = Image.new("RGBA", viz.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)
colors = [(255,0,0,110),(0,200,255,110),(255,220,0,110),(0,255,120,110),
          (200,0,255,110),(255,120,0,110),(0,100,255,110)]
for i, (name, pts) in enumerate(panels.items()):
    draw.polygon(pts, fill=colors[i % len(colors)], outline=(255,255,255,255))
Image.alpha_composite(viz, overlay).convert("RGB").save(f"{stem}_polygons_imgmask.png")
print(f"Saved {stem}_polygons_imgmask.png")

with open(f"{stem}_panels_imgmask.json", "w") as f:
    json.dump(panels, f, indent=2)
