"""Split a binary foreground mask into connected components, save each as
an alpha PNG composited with the original page image.

Each component becomes its own fg layer in the animation, independently
animatable. Components that are too small (< 1000 px) are dropped as noise.

Usage:  python3 gen_fg_components.py <page.png> <fg_mask.png> [prefix]

Output:
  <prefix>_c0.png, <prefix>_c1.png, ...     RGBA per-component images
  <prefix>_components.json                  metadata per component
"""
import json, pathlib, sys
import cv2
import numpy as np
from PIL import Image

src_page = pathlib.Path(sys.argv[1])
src_mask = pathlib.Path(sys.argv[2])
prefix = sys.argv[3] if len(sys.argv) > 3 else src_page.stem + "_fg"
out_dir = src_page.parent

MIN_AREA = 10000   # drop specks (anything smaller is noise, not a character)

page = Image.open(src_page).convert("RGBA")
mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
if mask.shape[:2] != (page.height, page.width):
    mask = cv2.resize(mask, (page.width, page.height), interpolation=cv2.INTER_NEAREST)
_, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

components = []
page_arr = np.array(page)  # HxWx4
for i in range(1, n):   # skip 0 = background
    x, y, w, h, area = stats[i]
    if area < MIN_AREA:
        continue
    comp_idx = len(components)
    # Alpha mask: 255 where this component, 0 elsewhere
    comp_alpha = np.where(labels == i, 255, 0).astype(np.uint8)
    comp_rgba = page_arr.copy()
    comp_rgba[..., 3] = comp_alpha
    comp_path = out_dir / f"{prefix}_c{comp_idx}.png"
    Image.fromarray(comp_rgba).save(comp_path)
    components.append({
        "id": comp_idx,
        "file": comp_path.name,
        "area": int(area),
        "centroid": [float(centroids[i][0]), float(centroids[i][1])],
        "bbox": [int(x), int(y), int(w), int(h)],
    })
    print(f"  c{comp_idx}: area={area:6d}  "
          f"centroid=({centroids[i][0]:6.1f}, {centroids[i][1]:6.1f})  "
          f"bbox=({x}, {y}, {w}, {h})  → {comp_path.name}")

meta_path = out_dir / f"{prefix}_components.json"
meta_path.write_text(json.dumps(components, indent=2))
print(f"\nSplit into {len(components)} components")
print(f"Metadata: {meta_path}")
