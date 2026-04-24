"""Hybrid panel detection v2 — LLM for topology, visual snap for each GUTTER
(not each panel edge), flood-fill from seeds to extract polygon shapes.

Pipeline:
  1. LLM gives panels { seed, bbox } — rough but topologically correct.
  2. From bbox adjacency, infer which pairs of panels share a gutter.
  3. For each inferred gutter, scan a wide band around the LLM's guess for the
     strongest bright/gradient line. Fit its equation. If nothing convincing,
     fall back to LLM's midline.
  4. Draw every gutter (snapped or fallback) as a thick wall. Plus page edges.
  5. Flood-fill from each seed. Contour → polygon.
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

# -------- Stage 1: LLM --------
cache = pathlib.Path(f"{stem}_seeds.json")
if cache.exists():
    seeds_data = json.loads(cache.read_text())
else:
    client = OpenAI(api_key=pathlib.Path.home().joinpath("Documents/openai_api_key.txt").read_text().strip())
    b64 = base64.b64encode(src.read_bytes()).decode()
    prompt = f"""Identify every comic panel on this {W}x{H} manga page in reading
order (top-to-bottom, left-to-right). For each panel, return:
  - "seed": one (x, y) pixel well INSIDE the panel interior, not near any edge
  - "bbox": the rough axis-aligned [x1,y1,x2,y2] of the panel. Conservative
     is better — stay INSIDE actual panel edges. If panel bleeds to a page
     edge, use 0 or {W}/{H}.
JSON only: {{"panels":[{{"name":"p1","seed":[x,y],"bbox":[x1,y1,x2,y2]}}, ...]}}"""
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

panels_llm = [(p["name"], tuple(p["seed"]), tuple(p["bbox"])) for p in seeds_data["panels"]]
print(f"{len(panels_llm)} panels")
for n, s, b in panels_llm:
    print(f"  {n}: seed={s} bbox={b}")

# -------- Stage 2: Infer shared gutters --------
# Two panels share a gutter if their bboxes are "edge-adjacent" in the
# horizontal or vertical axis.
# Direct-adjacency only: a pair shares a gutter only if their bbox gap along
# the adjacency axis is small (say under 40px). Otherwise there's another
# panel between them and the "gutter" we'd infer is spurious.
MAX_GAP = 40
gutters = []  # each: ("horiz"|"vert", y_or_x_guess, span_lo, span_hi, between=(nameA,nameB))
for i in range(len(panels_llm)):
    for j in range(i + 1, len(panels_llm)):
        ni, _, (xi1,yi1,xi2,yi2) = panels_llm[i]
        nj, _, (xj1,yj1,xj2,yj2) = panels_llm[j]
        x_overlap = min(xi2, xj2) - max(xi1, xj1)
        y_overlap = min(yi2, yj2) - max(yi1, yj1)
        # A above B
        if x_overlap > 30 and 0 <= (yj1 - yi2) <= MAX_GAP:
            y = (yi2 + yj1) // 2
            gutters.append(("horiz", y, max(xi1,xj1), min(xi2,xj2), (ni, nj)))
        # B above A
        elif x_overlap > 30 and 0 <= (yi1 - yj2) <= MAX_GAP:
            y = (yj2 + yi1) // 2
            gutters.append(("horiz", y, max(xi1,xj1), min(xi2,xj2), (ni, nj)))
        # A left of B
        elif y_overlap > 30 and 0 <= (xj1 - xi2) <= MAX_GAP:
            x = (xi2 + xj1) // 2
            gutters.append(("vert", x, max(yi1,yj1), min(yi2,yj2), (ni, nj)))
        # B left of A
        elif y_overlap > 30 and 0 <= (xi1 - xj2) <= MAX_GAP:
            x = (xj2 + xi1) // 2
            gutters.append(("vert", x, max(yi1,yj1), min(yi2,yj2), (ni, nj)))

print(f"\n{len(gutters)} inferred shared gutters")

# -------- Stage 3: Snap each gutter to visual evidence --------
# For each gutter, scan a wide band (±SCAN_MARGIN) for the strongest bright
# + gradient line, fit a line equation, and use it as the wall.
SCAN_MARGIN = 180
SAMPLE_STEP = 8

gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
grad = cv2.magnitude(gx, gy)
grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Boundary evidence: bright pixels + strong gradient pixels
_, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
_, grad_t = cv2.threshold(grad, 80, 255, cv2.THRESH_BINARY)
evidence = cv2.bitwise_or(bright, grad_t)

def snap_horizontal_gutter(y_guess, x_lo, x_hi):
    """Find best horizontal line y = m*x + b near y_guess between x_lo..x_hi.
    Returns (m, b) or None if nothing convincing."""
    y_top = max(0, y_guess - SCAN_MARGIN)
    y_bot = min(H, y_guess + SCAN_MARGIN)
    # For each sampled column, find the brightest row within the band.
    samples = []
    for x in range(max(0, x_lo), min(W, x_hi), SAMPLE_STEP):
        # Prefer bright pixels; require local darkness above/below for context
        col_evidence = evidence[y_top:y_bot, x]
        if col_evidence.sum() < 255 * 3:  # need at least 3 bright pixels
            continue
        # find the centroid row of the brightest contiguous run
        wh = np.where(col_evidence > 0)[0]
        if len(wh) == 0:
            continue
        # pick the run whose center is closest to y_guess - y_top
        target = y_guess - y_top
        # find the sample closest to the target
        best_idx = wh[np.argmin(np.abs(wh - target))]
        y_abs = y_top + int(best_idx)
        # context: must have darker panel content both above and below
        above = gray[max(0, y_abs-25):max(0, y_abs-10), x]
        below = gray[min(H, y_abs+10):min(H, y_abs+25), x]
        if len(above) and len(below) and above.mean() < 150 and below.mean() < 150:
            samples.append((x, y_abs))
    if len(samples) < 8:
        return None
    xs = np.array([s[0] for s in samples], float)
    ys = np.array([s[1] for s in samples], float)
    for _ in range(5):
        m, b = np.polyfit(xs, ys, 1)
        resid = ys - (m*xs + b)
        sd = resid.std()
        if sd < 0.5: break
        keep = np.abs(resid) < 1.5 * sd
        if keep.sum() < 8 or keep.all(): break
        xs, ys = xs[keep], ys[keep]
    m, b = np.polyfit(xs, ys, 1)
    return float(m), float(b), len(xs)

def snap_vertical_gutter(x_guess, y_lo, y_hi):
    """Find best vertical-ish line x = m*y + b near x_guess between y_lo..y_hi."""
    x_left = max(0, x_guess - SCAN_MARGIN)
    x_right = min(W, x_guess + SCAN_MARGIN)
    samples = []
    for y in range(max(0, y_lo), min(H, y_hi), SAMPLE_STEP):
        row_evidence = evidence[y, x_left:x_right]
        if row_evidence.sum() < 255 * 3:
            continue
        wh = np.where(row_evidence > 0)[0]
        if len(wh) == 0:
            continue
        target = x_guess - x_left
        best_idx = wh[np.argmin(np.abs(wh - target))]
        x_abs = x_left + int(best_idx)
        left = gray[y, max(0, x_abs-25):max(0, x_abs-10)]
        right = gray[y, min(W, x_abs+10):min(W, x_abs+25)]
        if len(left) and len(right) and left.mean() < 150 and right.mean() < 150:
            samples.append((x_abs, y))
    if len(samples) < 8:
        return None
    xs = np.array([s[0] for s in samples], float)
    ys = np.array([s[1] for s in samples], float)
    for _ in range(5):
        m, b = np.polyfit(ys, xs, 1)
        resid = xs - (m*ys + b)
        sd = resid.std()
        if sd < 0.5: break
        keep = np.abs(resid) < 1.5 * sd
        if keep.sum() < 8 or keep.all(): break
        xs, ys = xs[keep], ys[keep]
    m, b = np.polyfit(ys, xs, 1)
    return float(m), float(b), len(xs)

# -------- Stage 4: Build boundary from snapped gutters --------
boundary = np.zeros((H, W), np.uint8)
# Page edges
boundary[:3, :] = 255; boundary[-3:, :] = 255
boundary[:, :3] = 255; boundary[:, -3:] = 255

WALL_THICKNESS = 5

for kind, v_guess, a_lo, a_hi, between in gutters:
    if kind == "horiz":
        fit = snap_horizontal_gutter(v_guess, a_lo, a_hi)
        if fit is None:
            # Fall back to LLM guess
            yL = yR = v_guess
            print(f"  gutter {between}: FALLBACK y={v_guess}")
        else:
            m, b, n = fit
            yL = int(m * 0 + b); yR = int(m * (W-1) + b)
            print(f"  gutter {between}: snap h y(0)={yL} y({W-1})={yR} (n={n})")
        # Draw full-width line (extend to page edges so adjacent panels share it)
        cv2.line(boundary, (0, yL), (W-1, yR), 255, WALL_THICKNESS)
    else:
        fit = snap_vertical_gutter(v_guess, a_lo, a_hi)
        if fit is None:
            xT = xB = v_guess
            print(f"  gutter {between}: FALLBACK x={v_guess}")
        else:
            m, b, n = fit
            xT = int(m * 0 + b); xB = int(m * (H-1) + b)
            print(f"  gutter {between}: snap v x(0)={xT} x({H-1})={xB} (n={n})")
        cv2.line(boundary, (xT, 0), (xB, H-1), 255, WALL_THICKNESS)

cv2.imwrite(f"_{stem}_boundary_v2.png", boundary)

# -------- Stage 5: Flood fill from each seed --------
panel_mask = cv2.bitwise_not(boundary)
panels = {}
for name, (sx, sy), _ in panels_llm:
    sx, sy = int(sx), int(sy)
    if panel_mask[sy, sx] == 0:
        # nudge to nearest interior pixel
        found = False
        for r in range(3, 80, 3):
            ys = slice(max(0,sy-r), min(H,sy+r+1))
            xs = slice(max(0,sx-r), min(W,sx+r+1))
            region = panel_mask[ys, xs]
            if (region > 0).any():
                local = np.argwhere(region > 0)
                ly, lx = sy - max(0,sy-r), sx - max(0,sx-r)
                dists = (local[:,0]-ly)**2 + (local[:,1]-lx)**2
                best = local[dists.argmin()]
                sy, sx = best[0] + max(0,sy-r), best[1] + max(0,sx-r)
                found = True; break
        if not found:
            print(f"  {name}: seed stuck")
            continue
    ff_mask = np.zeros((H+2, W+2), np.uint8)
    ff = panel_mask.copy()
    cv2.floodFill(ff, ff_mask, (sx, sy), 128, 0, 0, flags=4 | (255 << 8))
    filled = (ff_mask[1:-1,1:-1] > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue
    cnt = max(cnts, key=cv2.contourArea)
    perim = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.004 * perim, True)
    panels[name] = [(int(p[0][0]), int(p[0][1])) for p in approx]

print("\nFinal panels:")
for name, pts in panels.items():
    print(f"  {name}: {len(pts)} verts")

viz = Image.open(src).convert("RGBA")
overlay = Image.new("RGBA", viz.size, (0,0,0,0))
draw = ImageDraw.Draw(overlay)
colors = [(255,0,0,110),(0,200,255,110),(255,220,0,110),(0,255,120,110),
          (200,0,255,110),(255,120,0,110),(0,100,255,110)]
for i, (name, pts) in enumerate(panels.items()):
    draw.polygon(pts, fill=colors[i % len(colors)], outline=(255,255,255,255))
for name, (x, y), _ in panels_llm:
    draw.ellipse([x-6, y-6, x+6, y+6], fill=(255,255,255,255), outline=(0,0,0,255))
Image.alpha_composite(viz, overlay).convert("RGB").save(f"{stem}_polygons_v2.png")
print(f"Saved {stem}_polygons_v2.png")
with open(f"{stem}_panels_v2.json","w") as f: json.dump(panels, f, indent=2)
