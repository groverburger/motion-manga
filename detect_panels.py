"""Detect manga panels as polygon contours using OpenCV.

Standard approach (from pvnieo/Manga-Panel-Extractor and similar):
  1. Convert to grayscale
  2. Add a thick black border so panels are closed at page edges
  3. Threshold to isolate gutter whites
  4. findContours on the inverted image (panels = external contours)
  5. approxPolyDP each contour to get its polygon corners
  6. Filter by area

This gives real angled polygons matching the panel shapes.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw
import json
import sys
import os

src = sys.argv[1] if len(sys.argv) > 1 else "page1.png"
stem = os.path.splitext(os.path.basename(src))[0]
print(f"=== {src} ===")

img_bgr = cv2.imread(src)
H, W = img_bgr.shape[:2]
print(f"Image: {W}x{H}")

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Add a WHITE border so: after threshold (gutters=white, panels=black) and
# inversion, the border becomes black and isolates panels from the outside.
BORDER = 20
bordered = cv2.copyMakeBorder(gray, BORDER, BORDER, BORDER, BORDER,
                              cv2.BORDER_CONSTANT, value=255)

# Blur for noise reduction before thresholding.
blur = cv2.GaussianBlur(bordered, (5, 5), 0)

# Threshold: pixels brighter than 210 are gutter/background.
_, th = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY)

# Dilate isotropically so gutters stay connected where panel art encroaches.
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
th = cv2.dilate(th, kernel_dilate, iterations=2)

# Gentle OPENING: break thin bridges between interior bright patches and the
# gutter network without breaking the gutters themselves. Smaller than gutter
# thickness (which is ~8-10px after dilation).
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_open, iterations=1)

cv2.imwrite(f"_{stem}_thresh.png", th)

# Flood-fill from the white border inward. Since the border connects to all
# gutters (they're both white), flooding from (0,0) with a marker will mark:
#   border + all gutters -> marker color
# Panel interiors remain untouched because they're black (disconnected).
flood = th.copy()
h, w = flood.shape
mask = np.zeros((h + 2, w + 2), np.uint8)
# fill the entire white gutter-network with value 128 starting at (0,0)
cv2.floodFill(flood, mask, (0, 0), 128)

# Now: 0=panel interior, 128=gutter+border, 255=any still-white island (text,
# bright highlights surrounded by panel content — these are NOT panels).
# Any pixel that is NOT marked as gutter-network (128) belongs to a panel.
# That includes dark panel interiors (still 0) AND bright islands inside a
# panel that weren't connected to the gutter network (text bubbles, highlights
# — still 255). This gives solid panel blobs, no internal notches.
panels_mask = (flood != 128).astype(np.uint8) * 255
cv2.imwrite(f"_{stem}_filled.png", panels_mask)

contours, _ = cv2.findContours(panels_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

total_area = (W + 2*BORDER) * (H + 2*BORDER)
panels_raw = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    pct = area / total_area
    if pct < 0.02 or pct > 0.80:
        continue
    # Simplify the contour to its essential corners. No convex hull — real
    # panels can be concave (e.g. a character figure "breaking" the panel
    # border and extending into the next panel's area).
    perim = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.008 * perim, True)
    pts = [(int(p[0][0]) - BORDER, int(p[0][1]) - BORDER) for p in approx]
    # Clip to page bounds
    pts = [(max(0, min(W, x)), max(0, min(H, y))) for x, y in pts]
    panels_raw.append((area, pts))

# Sort by area descending, then by top-y to order reading-wise top→bottom, left→right
panels_raw.sort(key=lambda p: (-p[0]))
for i, (area, pts) in enumerate(panels_raw):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    print(f"  Panel {i}: area={int(area)}  verts={len(pts)}  centroid=({cx:.0f},{cy:.0f})")

# Sort panels into reading order: group into "rows" by y-centroid, then
# sort left-to-right within each row. Row cutoff = half the median panel height.
def centroid(pts):
    return (sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))

raw = [(centroid(pts), pts) for _, pts in panels_raw]
heights = []
for _, pts in raw:
    ys = [p[1] for p in pts]
    heights.append(max(ys) - min(ys))
row_gap = (sorted(heights)[len(heights) // 2]) // 2 if heights else 200

raw.sort(key=lambda r: r[0][1])  # by y
rows = []
for (cx, cy), pts in raw:
    if rows and abs(cy - rows[-1][0][0][1]) < row_gap:
        rows[-1].append(((cx, cy), pts))
    else:
        rows.append([((cx, cy), pts)])
for row in rows:
    row.sort(key=lambda r: r[0][0])  # by x within row

panels = {}
i = 1
for row in rows:
    for _, pts in row:
        panels[f"p{i}"] = pts
        i += 1

print(f"\n{len(panels)} panels in reading order:")
for k, pts in panels.items():
    print(f"  {k}: {pts}")

# Visualize
viz = Image.open(src).convert("RGBA")
overlay = Image.new("RGBA", viz.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)
colors = {
    "p1": (255, 0,   0,   110),
    "p2": (0,   200, 255, 110),
    "p3": (255, 220, 0,   110),
    "p4": (0,   255, 120, 110),
    "p5": (200, 0,   255, 110),
}
for name, pts in panels.items():
    draw.polygon(pts, fill=colors[name], outline=(255, 255, 255, 255))
out = Image.alpha_composite(viz, overlay)
out.convert("RGB").save(f"{stem}_polygons.png")
print(f"Saved {stem}_polygons.png")

with open(f"{stem}_panels.json", "w") as f:
    json.dump(panels, f, indent=2)
print(f"Saved {stem}_panels.json")
