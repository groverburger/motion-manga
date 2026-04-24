"""Hybrid panel detection: LLM for structure, classical CV for edge snapping.

1. Vision LLM identifies each panel and returns a rough axis-aligned bbox.
2. For each non-page-edge side of each bbox, scan a narrow band around the
   rough edge to find the true gutter line, which may be angled. Fit a line.
3. Compute final polygon corners as intersections of adjacent edges.

This gives us the LLM's robust topology understanding plus pixel-accurate
geometry from classical CV, without either's weaknesses.
"""
import base64, json, pathlib, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
from openai import OpenAI

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "page1.png")
stem = src.stem
img_gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
H, W = img_gray.shape
print(f"=== {src} ({W}x{H}) ===")

# ------- Stage 1: LLM for structure -------
llm_cache = pathlib.Path(f"{stem}_panels_llm.json")
if llm_cache.exists():
    print("Using cached LLM response")
    data = json.loads(llm_cache.read_text())
else:
    client = OpenAI(api_key=pathlib.Path.home().joinpath("Documents/openai_api_key.txt").read_text().strip())
    b64 = base64.b64encode(src.read_bytes()).decode()
    prompt = f"""Analyze this {W}x{H} manga page. Return the rough axis-aligned
bounding box of each panel in reading order (top→bottom, left→right).

Just rectangles are fine — I'll refine the exact angles afterwards. What I need
from you is the correct count of panels and their approximate positions.

JSON only: {{"panels": [{{"name":"p1","bbox":[x1,y1,x2,y2]}}, ...]}}"""
    resp = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    llm_cache.write_text(json.dumps(data, indent=2))

def to_bbox(p):
    if "bbox" in p:
        return tuple(p["bbox"])
    # Fall back: convert vertex list to axis-aligned bbox
    xs = [v[0] for v in p["vertices"]]
    ys = [v[1] for v in p["vertices"]]
    return (min(xs), min(ys), max(xs), max(ys))

bboxes = [(p["name"], to_bbox(p)) for p in data["panels"]]
print(f"{len(bboxes)} panels from LLM")
for name, bb in bboxes:
    print(f"  {name}: {bb}")

# ------- Stage 2: Edge snapping -------
EDGE_SNAP_MARGIN = 12   # a rough edge this close to the page edge = "bleeds"
SCAN_MARGIN = 80        # search ±this many px around the rough edge
SAMPLE_STEP = 12        # sampling density along the edge
MIN_BRIGHT = 200        # gutter pixel threshold
DARK_CONTEXT = 130      # rows/cols on each side of a gutter must be this dark

def is_page_edge(v, limit):
    return v <= EDGE_SNAP_MARGIN or v >= limit - EDGE_SNAP_MARGIN

def snap_horizontal(y_rough, x_start, x_end):
    """Find the best horizontal(ish) gutter line in y ∈ [y_rough-margin, y_rough+margin]
    between x_start and x_end. Return (m, b) such that y = m*x + b, or None."""
    if is_page_edge(y_rough, H):
        return None  # bleeds — don't snap
    y_lo = max(0, y_rough - SCAN_MARGIN)
    y_hi = min(H, y_rough + SCAN_MARGIN)
    samples = []
    for x in range(max(0, x_start), min(W, x_end), SAMPLE_STEP):
        col = img_gray[y_lo:y_hi, x]
        # Only accept a gutter pixel that's bright AND has dark context above+below
        order = np.argsort(col)[::-1]
        for idx in order[:5]:
            if col[idx] < MIN_BRIGHT:
                break
            y_abs = y_lo + int(idx)
            above = img_gray[max(0, y_abs-22):max(0, y_abs-10), x]
            below = img_gray[min(H, y_abs+10):min(H, y_abs+22), x]
            if len(above) and len(below) and above.mean() < DARK_CONTEXT and below.mean() < DARK_CONTEXT:
                samples.append((x, y_abs))
                break
    if len(samples) < 5:
        return None
    # Robust line fit with outlier rejection
    xs = np.array([s[0] for s in samples], dtype=float)
    ys = np.array([s[1] for s in samples], dtype=float)
    for _ in range(4):
        m, b = np.polyfit(xs, ys, 1)
        resid = ys - (m * xs + b)
        sd = resid.std()
        if sd < 0.5: break
        keep = np.abs(resid) < 1.8 * sd
        if keep.sum() < 5 or keep.all(): break
        xs, ys = xs[keep], ys[keep]
    m, b = np.polyfit(xs, ys, 1)
    return float(m), float(b)

def snap_vertical(x_rough, y_start, y_end):
    """Find best vertical(ish) gutter. Return (m, b) such that x = m*y + b."""
    if is_page_edge(x_rough, W):
        return None
    x_lo = max(0, x_rough - SCAN_MARGIN)
    x_hi = min(W, x_rough + SCAN_MARGIN)
    samples = []
    for y in range(max(0, y_start), min(H, y_end), SAMPLE_STEP):
        row = img_gray[y, x_lo:x_hi]
        order = np.argsort(row)[::-1]
        for idx in order[:5]:
            if row[idx] < MIN_BRIGHT:
                break
            x_abs = x_lo + int(idx)
            left = img_gray[y, max(0, x_abs-22):max(0, x_abs-10)]
            right = img_gray[y, min(W, x_abs+10):min(W, x_abs+22)]
            if len(left) and len(right) and left.mean() < DARK_CONTEXT and right.mean() < DARK_CONTEXT:
                samples.append((x_abs, y))
                break
    if len(samples) < 5:
        return None
    xs = np.array([s[0] for s in samples], dtype=float)
    ys = np.array([s[1] for s in samples], dtype=float)
    for _ in range(4):
        m, b = np.polyfit(ys, xs, 1)
        resid = xs - (m * ys + b)
        sd = resid.std()
        if sd < 0.5: break
        keep = np.abs(resid) < 1.8 * sd
        if keep.sum() < 5 or keep.all(): break
        xs, ys = xs[keep], ys[keep]
    m, b = np.polyfit(ys, xs, 1)
    return float(m), float(b)

# Snap each panel's edges
snapped = []
for name, (x1, y1, x2, y2) in bboxes:
    top    = snap_horizontal(y1, x1, x2)       # (m, b) or None (bleed)
    bottom = snap_horizontal(y2, x1, x2)
    left   = snap_vertical(x1, y1, y2)
    right  = snap_vertical(x2, y1, y2)
    snapped.append({
        "name": name, "bbox": (x1, y1, x2, y2),
        "top": top, "bottom": bottom, "left": left, "right": right,
    })
    print(f"  {name} snap: top={top} bot={bottom} left={left} right={right}")

# Intersect adjacent edges to compute polygon corners.
# A horizontal line is y = mh*x + bh. A vertical line is x = mv*y + bv.
# Intersection:
#   y = mh*(mv*y + bv) + bh
#   y*(1 - mh*mv) = mh*bv + bh
#   y = (mh*bv + bh) / (1 - mh*mv)
def intersect(h, v, fallback_x, fallback_y):
    """Intersection of horizontal-ish line h=(mh,bh) and vertical-ish line v=(mv,bv).
    If either is None (bleed), fall back to fallback coordinates."""
    if h is None and v is None:
        return (fallback_x, fallback_y)
    if h is None:
        # Edge is vertical only; y = fallback_y
        mv, bv = v
        return (mv * fallback_y + bv, fallback_y)
    if v is None:
        mh, bh = h
        return (fallback_x, mh * fallback_x + bh)
    mh, bh = h
    mv, bv = v
    denom = 1 - mh * mv
    if abs(denom) < 1e-6:
        return (fallback_x, fallback_y)
    y = (mh * bv + bh) / denom
    x = mv * y + bv
    return (x, y)

panels = {}
for s in snapped:
    x1, y1, x2, y2 = s["bbox"]
    tl = intersect(s["top"],    s["left"],  x1, y1)
    tr = intersect(s["top"],    s["right"], x2, y1)
    br = intersect(s["bottom"], s["right"], x2, y2)
    bl = intersect(s["bottom"], s["left"],  x1, y2)
    pts = [(int(round(px)), int(round(py))) for px, py in (tl, tr, br, bl)]
    # Clamp to page bounds
    pts = [(max(0, min(W, x)), max(0, min(H, y))) for x, y in pts]
    panels[s["name"]] = pts

print("\nFinal panels:")
for name, pts in panels.items():
    print(f"  {name}: {pts}")

# Visualize
viz = Image.open(src).convert("RGBA")
overlay = Image.new("RGBA", viz.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)
colors = [(255,0,0,110),(0,200,255,110),(255,220,0,110),(0,255,120,110),
          (200,0,255,110),(255,120,0,110),(0,100,255,110)]
for i, (name, pts) in enumerate(panels.items()):
    draw.polygon(pts, fill=colors[i % len(colors)], outline=(255,255,255,255))
Image.alpha_composite(viz, overlay).convert("RGB").save(f"{stem}_polygons_hybrid.png")
print(f"\nSaved {stem}_polygons_hybrid.png")

with open(f"{stem}_panels_hybrid.json", "w") as f:
    json.dump(panels, f, indent=2)
