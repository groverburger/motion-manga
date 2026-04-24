"""Seeded flood-fill panel detection.

Simpler than watershed and closer to the user's original intuition:
  1. Build a "boundary" binary mask where 1 = panel border, 0 = panel interior
  2. LLM provides one interior seed point per panel
  3. For each seed, flood-fill the 0-valued connected component containing it
  4. Each flood = that panel's region. Contour it for the polygon.

Naturally handles concavity (region growing follows actual shape). Seeds make
it robust to faint gutters — as long as SOME boundary signal exists between
panels, the floods from different seeds stay separate.
"""
import base64, json, pathlib, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
from openai import OpenAI

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "page1.png")
stem = src.stem
img_bgr = cv2.imread(str(src))
H, W = img_bgr.shape[:2]
print(f"=== {src} ({W}x{H}) ===")

# -------- Stage 1: LLM for seeds --------
seeds_cache = pathlib.Path(f"{stem}_seeds.json")
if seeds_cache.exists():
    print("Using cached LLM seeds")
    seeds_data = json.loads(seeds_cache.read_text())
else:
    client = OpenAI(api_key=pathlib.Path.home().joinpath("Documents/openai_api_key.txt").read_text().strip())
    b64 = base64.b64encode(src.read_bytes()).decode()
    prompt = f"""Identify every comic panel on this {W}x{H} manga page in reading order
(top-to-bottom, left-to-right).

For each panel, return:
  - "seed": one (x, y) pixel clearly INSIDE the panel (interior, not near any edge)
  - "bbox": the rough axis-aligned bounding box [x1, y1, x2, y2] of the panel.
    Conservative is better than too generous — stay INSIDE the actual panel edges
    rather than overlapping into adjacent panels. If the panel bleeds to a page
    edge, use 0 or {W}/{H}.

JSON only: {{"panels": [{{"name":"p1","seed":[x,y],"bbox":[x1,y1,x2,y2]}}, ...]}}"""
    resp = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        response_format={"type": "json_object"},
    )
    seeds_data = json.loads(resp.choices[0].message.content)
    seeds_cache.write_text(json.dumps(seeds_data, indent=2))

seeds = [(p["name"], tuple(p["seed"]), tuple(p.get("bbox", (0, 0, W, H))))
         for p in seeds_data["panels"]]
print(f"{len(seeds)} panels from LLM")
for name, seed, bbox in seeds:
    print(f"  {name}: seed={seed}  bbox={bbox}")

# -------- Stage 2: Build boundary mask --------
# A pixel is "boundary" if EITHER:
#   - It's bright (part of a white gutter), OR
#   - It has strong LONG-RANGE gradient (panel border transition, smoothed
#     enough that internal line art doesn't count)
# Heavy bilateral filtering of the source kills internal-art gradient while
# preserving sharp gutter/border transitions.

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Multiple passes of bilateral filter → strong smoothing, edge-preserving.
smooth = gray
for _ in range(3):
    smooth = cv2.bilateralFilter(smooth, d=15, sigmaColor=80, sigmaSpace=80)
cv2.imwrite(f"_{stem}_smooth.png", smooth)

# (a) bright-gutter signal: threshold with a moderate cutoff.
_, bright = cv2.threshold(smooth, 180, 255, cv2.THRESH_BINARY)

# (b) gradient of the smoothed image. Post-smoothing, only major transitions
# survive — no more interior ink noise.
gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=5)
gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=5)
grad = cv2.magnitude(gx, gy)
grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, grad_strong = cv2.threshold(grad, 70, 255, cv2.THRESH_BINARY)

boundary = cv2.bitwise_or(bright, grad_strong)

# Thicken boundaries to make sure floods can't sneak through 1-px gaps.
boundary = cv2.dilate(boundary,
                      cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                      iterations=2)
# Also close small holes in the boundary network (bridges gaps where a
# character breaks through a gutter). Horizontal + vertical separately to
# avoid fusing adjacent gutters together.
boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)))

# Filter visual boundary by connected-component SHAPE:
# - True gutters are long, thin, elongated LINES (major axis >> minor axis)
# - Interior noise (text bubbles, highlights) is ROUND blobs
# Use connectedComponentsWithStats and keep only elongated or large components.
n, labels, stats, _ = cv2.connectedComponentsWithStats(boundary, connectivity=8)
filtered = np.zeros_like(boundary)
for i in range(1, n):
    x, y, w, h, area = stats[i]
    # Elongation = max extent / min extent in pixels. Lines have high values.
    elongation = max(w, h) / max(1, min(w, h))
    touches_edge = (x < 3 or y < 3 or x + w > W - 3 or y + h > H - 3)
    if elongation > 3.5 or (touches_edge and area > 200):
        filtered[labels == i] = 255
visual_boundary = filtered

# LLM-inferred walls as a FALLBACK when visual signal is missing (e.g., gutters
# that are only a tonal transition with no bright/gradient cue).
llm_walls = np.zeros_like(boundary)
for i, (_, _, bb_i) in enumerate(seeds):
    for j, (_, _, bb_j) in enumerate(seeds):
        if i >= j:
            continue
        (xi1, yi1, xi2, yi2) = bb_i
        (xj1, yj1, xj2, yj2) = bb_j
        if yi2 <= yj1 and not (xi2 <= xj1 or xj2 <= xi1):
            y_wall = (yi2 + yj1) // 2
            cv2.line(llm_walls, (max(xi1,xj1), y_wall), (min(xi2,xj2), y_wall), 255, 1)
        elif yj2 <= yi1 and not (xi2 <= xj1 or xj2 <= xi1):
            y_wall = (yj2 + yi1) // 2
            cv2.line(llm_walls, (max(xi1,xj1), y_wall), (min(xi2,xj2), y_wall), 255, 1)
        elif xi2 <= xj1 and not (yi2 <= yj1 or yj2 <= yi1):
            x_wall = (xi2 + xj1) // 2
            cv2.line(llm_walls, (x_wall, max(yi1,yj1)), (x_wall, min(yi2,yj2)), 255, 1)
        elif xj2 <= xi1 and not (yi2 <= yj1 or yj2 <= yi1):
            x_wall = (xj2 + xi1) // 2
            cv2.line(llm_walls, (x_wall, max(yi1,yj1)), (x_wall, min(yi2,yj2)), 255, 1)
# Only keep LLM walls where NO visual boundary exists nearby (within 40px).
# Where visual signal is strong, trust that. Where it's missing, LLM fills in.
visual_near = cv2.dilate(visual_boundary,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
llm_fallback = cv2.bitwise_and(llm_walls, cv2.bitwise_not(visual_near))
llm_fallback = cv2.dilate(llm_fallback,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                           iterations=2)

# Page edge wall.
page_edge = np.zeros_like(boundary)
page_edge[:3, :] = 255; page_edge[-3:, :] = 255
page_edge[:, :3] = 255; page_edge[:, -3:] = 255

boundary = cv2.bitwise_or(visual_boundary, llm_fallback)
boundary = cv2.bitwise_or(boundary, page_edge)

cv2.imwrite(f"_{stem}_boundary.png", boundary)

# -------- Stage 3: Flood-fill from each seed --------
# panel_mask is 255 inside panels, 0 on boundaries.
panel_mask = cv2.bitwise_not(boundary)

panels = {}
for name, (x, y), _bbox in seeds:
    # Check if seed actually sits on a non-boundary pixel. If the LLM put it
    # on a dark character body that happens to be flagged as boundary, nudge
    # to a nearby panel-interior pixel.
    if panel_mask[y, x] == 0:
        # Search outward for a nearby panel pixel
        found = False
        for r in range(3, 60, 3):
            ys, xs = np.ogrid[max(0, y-r):min(H, y+r+1), max(0, x-r):min(W, x+r+1)]
            region = panel_mask[max(0,y-r):min(H,y+r+1), max(0,x-r):min(W,x+r+1)]
            if (region > 0).any():
                candidates = np.argwhere(region > 0)
                # pick closest to (y, x) in local coords
                cy, cx = y - max(0, y-r), x - max(0, x-r)
                dists = (candidates[:, 0] - cy) ** 2 + (candidates[:, 1] - cx) ** 2
                best = candidates[dists.argmin()]
                y = best[0] + max(0, y-r)
                x = best[1] + max(0, x-r)
                found = True
                break
        if not found:
            print(f"  {name}: seed stuck on boundary, skipping")
            continue

    # Flood fill the panel_mask from this seed. Mark the filled region with a
    # unique color, then extract.
    ff = panel_mask.copy()
    # floodFill mask must be (H+2, W+2)
    ff_mask = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(ff, ff_mask, (int(x), int(y)), newVal=128,
                  loDiff=0, upDiff=0, flags=4 | (255 << 8))
    filled = (ff_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    cnt = max(contours, key=cv2.contourArea)
    perim = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.003 * perim, True)  # tight epsilon preserves concavity
    pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
    panels[name] = pts

print("\nPanels:")
for name, pts in panels.items():
    print(f"  {name}: {len(pts)} verts")

# -------- Stage 4: Visualize --------
viz = Image.open(src).convert("RGBA")
overlay = Image.new("RGBA", viz.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)
colors = [(255,0,0,110),(0,200,255,110),(255,220,0,110),(0,255,120,110),
          (200,0,255,110),(255,120,0,110),(0,100,255,110),(120,120,255,110)]
for i, (name, pts) in enumerate(panels.items()):
    draw.polygon(pts, fill=colors[i % len(colors)], outline=(255,255,255,255))
for i, (name, (x, y), _bbox) in enumerate(seeds):
    draw.ellipse([x-6, y-6, x+6, y+6], fill=(255,255,255,255), outline=(0,0,0,255))

Image.alpha_composite(viz, overlay).convert("RGB").save(f"{stem}_polygons_watershed.png")
print(f"\nSaved {stem}_polygons_watershed.png")

with open(f"{stem}_panels_watershed.json", "w") as f:
    json.dump(panels, f, indent=2)
