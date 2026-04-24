"""Panel detection v3 — Hough-based gutter line detection + LLM seeds.

Core idea: panel gutters are LONG straight line segments. Hough line transform
is the correct tool for detecting them directly, at any angle, robust against
short/local bright features.

Pipeline:
  1. Build a "gutter candidate" mask: bright pixels (white gutters) OR strong
     gradient transitions (panel borders). Thin to skeleton for Hough.
  2. HoughLinesP → all line segments of meaningful length.
  3. Merge collinear segments, filter to ones that span a big chunk of the
     page (panel gutters) vs. tiny noise (text bubble edges).
  4. Paint these as walls in the boundary mask. Add page edges.
  5. Use LLM seeds (one per panel) to flood-fill each panel region.
  6. Contour → polygon.
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
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
print(f"=== {src} ({W}x{H}) ===")

# -------- Stage 1: LLM seeds (reuse existing cache) --------
cache = pathlib.Path(f"{stem}_seeds.json")
if cache.exists():
    seeds_data = json.loads(cache.read_text())
else:
    client = OpenAI(api_key=pathlib.Path.home().joinpath("Documents/openai_api_key.txt").read_text().strip())
    b64 = base64.b64encode(src.read_bytes()).decode()
    prompt = f"""Identify every comic panel on this {W}x{H} manga page in reading order.
For each panel, return one (x, y) seed INSIDE it.
JSON: {{"panels":[{{"name":"p1","seed":[x,y]}}, ...]}}"""
    resp = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}},
        ]}],
        response_format={"type":"json_object"},
    )
    seeds_data = json.loads(resp.choices[0].message.content)
    cache.write_text(json.dumps(seeds_data, indent=2))

# Accept either old format (with bbox) or new (seeds only)
seeds = [(p["name"], tuple(p["seed"])) for p in seeds_data["panels"]]
print(f"{len(seeds)} seeds from LLM")

# -------- Stage 2: Build gutter-candidate mask --------
# Bright gutters
_, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Strong gradient transitions (Canny is calibrated for line-like edges)
canny = cv2.Canny(gray, 80, 200)

# Thicken so thin features are robust, then skeletonize for cleaner Hough input
mask = cv2.bitwise_or(bright, canny)
mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                  iterations=1)

cv2.imwrite(f"_{stem}_candidates.png", mask)

# -------- Stage 3: Hough line detection --------
# HoughLinesP with a much more demanding min length — real gutters span
# most of the page width or height.
min_line_len = int(min(H, W) * 0.50)
lines_raw = cv2.HoughLinesP(
    mask,
    rho=1,
    theta=np.pi / 360,
    threshold=120,
    minLineLength=min_line_len,
    maxLineGap=80,
)
print(f"HoughLinesP raw: {0 if lines_raw is None else len(lines_raw)} lines")

if lines_raw is None:
    lines_raw = np.zeros((0, 1, 4), dtype=np.int32)

# Filter to near-axis-aligned lines only. Panel gutters are near-horizontal
# (within 20° of horizontal) or near-vertical (within 20° of vertical).
# Highly oblique interior lines (action lines, building diagonals) are rejected.
def line_angle_deg(x1, y1, x2, y2):
    return abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

segments = []
for l in lines_raw:
    x1, y1, x2, y2 = l[0]
    a = line_angle_deg(x1, y1, x2, y2)
    # a is in [0, 180]. 0 or 180 = horizontal; 90 = vertical.
    near_horiz = a < 20 or a > 160
    near_vert = 70 < a < 110
    if near_horiz or near_vert:
        segments.append(tuple(l[0]))
print(f"After angle filter: {len(segments)} segments")

# -------- Stage 4: Merge collinear segments --------
# Two segments are collinear if their lines (extended to page) are close in
# both angle and perpendicular distance.
def segment_angle_length(s):
    x1, y1, x2, y2 = s
    return np.arctan2(y2 - y1, x2 - x1), np.hypot(x2 - x1, y2 - y1)

def merge_collinear(segments, angle_tol=np.radians(8), dist_tol=25):
    if not segments: return []
    # Represent each segment by (angle, perp_offset) in normal form.
    # Merge ones whose angle and perp_offset are close.
    clusters = []
    for s in segments:
        a, L = segment_angle_length(s)
        # normal form: ax + by = c, normalised
        dx = np.cos(a); dy = np.sin(a)
        nx, ny = -dy, dx  # normal to the line direction
        x1, y1, x2, y2 = s
        perp = nx * x1 + ny * y1
        placed = False
        for cl in clusters:
            ca, cperp = cl["angle"], cl["perp"]
            if abs(((a - ca + np.pi/2) % np.pi) - np.pi/2) < angle_tol and abs(perp - cperp) < dist_tol:
                cl["segments"].append(s)
                cl["length"] += L
                placed = True
                break
        if not placed:
            clusters.append({"angle": a, "perp": perp, "segments": [s], "length": L})
    # For each cluster, fit a single line through all endpoints
    merged = []
    for cl in clusters:
        pts = []
        for s in cl["segments"]:
            pts.append((s[0], s[1])); pts.append((s[2], s[3]))
        pts = np.array(pts, dtype=np.float32)
        # Fit a line via cv2.fitLine (least squares)
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        # Extend to page edges in the direction of the line
        if abs(vx) > abs(vy):  # more horizontal
            t1 = (0 - x0) / vx
            t2 = (W - 1 - x0) / vx
        else:
            t1 = (0 - y0) / vy
            t2 = (H - 1 - y0) / vy
        p1 = (x0 + t1 * vx, y0 + t1 * vy)
        p2 = (x0 + t2 * vx, y0 + t2 * vy)
        merged.append({"p1": p1, "p2": p2, "length": cl["length"],
                        "angle": cl["angle"], "n_segments": len(cl["segments"])})
    return merged

gutters = merge_collinear(segments)
# Sort by total length supporting the line — longest first
gutters.sort(key=lambda g: -g["length"])
print(f"Merged to {len(gutters)} unique lines")
for i, g in enumerate(gutters[:12]):
    print(f"  line {i}: p1={g['p1']}  p2={g['p2']}  len={g['length']:.0f}  segs={g['n_segments']}")

# Keep only lines with enough supporting segment length (filter noise)
min_support = max(min_line_len, int(min(H, W) * 0.35))
kept = [g for g in gutters if g["length"] > min_support]
print(f"Kept {len(kept)} lines after length filter (min={min_support})")

# -------- Stage 5: Build boundary from Hough lines --------
boundary = np.zeros((H, W), np.uint8)
boundary[:3, :] = 255; boundary[-3:, :] = 255
boundary[:, :3] = 255; boundary[:, -3:] = 255
for g in kept:
    (x1, y1), (x2, y2) = g["p1"], g["p2"]
    cv2.line(boundary, (int(x1), int(y1)), (int(x2), int(y2)), 255, 5)

cv2.imwrite(f"_{stem}_boundary_v3.png", boundary)

# -------- Stage 6: Flood fill from seeds --------
panel_mask = cv2.bitwise_not(boundary)
panels = {}
for name, (sx, sy) in seeds:
    sx, sy = int(sx), int(sy)
    if panel_mask[sy, sx] == 0:
        for r in range(3, 80, 3):
            ys, xs = slice(max(0,sy-r), min(H,sy+r+1)), slice(max(0,sx-r), min(W,sx+r+1))
            region = panel_mask[ys, xs]
            if (region > 0).any():
                local = np.argwhere(region > 0)
                ly, lx = sy - max(0,sy-r), sx - max(0,sx-r)
                dists = (local[:,0]-ly)**2 + (local[:,1]-lx)**2
                best = local[dists.argmin()]
                sy, sx = best[0] + max(0,sy-r), best[1] + max(0,sx-r)
                break
    ff_mask = np.zeros((H+2, W+2), np.uint8)
    ff = panel_mask.copy()
    cv2.floodFill(ff, ff_mask, (sx, sy), 128, 0, 0, flags=4 | (255 << 8))
    filled = (ff_mask[1:-1,1:-1] > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: continue
    cnt = max(cnts, key=cv2.contourArea)
    perim = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.004 * perim, True)
    panels[name] = [(int(p[0][0]), int(p[0][1])) for p in approx]

print("\nFinal panels:")
for name, pts in panels.items():
    print(f"  {name}: {len(pts)} verts")

# Visualize with Hough lines shown
viz = Image.open(src).convert("RGBA")
overlay = Image.new("RGBA", viz.size, (0,0,0,0))
draw = ImageDraw.Draw(overlay)
colors = [(255,0,0,110),(0,200,255,110),(255,220,0,110),(0,255,120,110),
          (200,0,255,110),(255,120,0,110),(0,100,255,110)]
for i, (name, pts) in enumerate(panels.items()):
    draw.polygon(pts, fill=colors[i % len(colors)], outline=(255,255,255,255))
# Draw the detected Hough gutter lines in bright green for diagnosis
for g in kept:
    (x1, y1), (x2, y2) = g["p1"], g["p2"]
    draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0, 255), width=2)
for name, (x, y) in seeds:
    draw.ellipse([x-6, y-6, x+6, y+6], fill=(255,255,255,255), outline=(0,0,0,255))
Image.alpha_composite(viz, overlay).convert("RGB").save(f"{stem}_polygons_v3.png")
print(f"Saved {stem}_polygons_v3.png")
with open(f"{stem}_panels_v3.json","w") as f: json.dump(panels, f, indent=2)
