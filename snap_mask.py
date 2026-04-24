"""Refine an image-gen panel mask by snapping its boundaries to real gutter
edges in the source page.

Pipeline:
  1. Extract each panel polygon from the mask (contour → approxPolyDP → 4-8 verts).
  2. Classify each polygon edge as either a "gutter edge" (interior to page)
     or a "page edge" (vertex on x∈{0,W} or y∈{0,H}).
  3. For each GUTTER edge, sample points along the edge. At each sample, scan
     perpendicular in the ORIGINAL image for the nearest strong gutter signal
     (bright pixel AND high gradient magnitude). Snap to it if found.
  4. Robust-fit a line through the snapped samples → this becomes the refined edge.
  5. Panel vertices are recomputed as intersections of adjacent refined edges
     (or kept as page-edge intersections unchanged).
"""
import sys, pathlib, json
import numpy as np
import cv2
from PIL import Image, ImageDraw

page = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "page1.png")
mask_path = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "page1_mask.png")

orig_bgr = cv2.imread(str(page))
orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
H, W = orig_gray.shape
m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
if m.shape != (H, W):
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
_, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

# ---- Extract initial panel polygons from the mask ----
contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
panels_raw = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < W * H * 0.01 or area > W * H * 0.9:
        continue
    perim = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.003 * perim, True)
    verts = [(int(p[0][0]), int(p[0][1])) for p in approx]
    panels_raw.append(verts)

print(f"Extracted {len(panels_raw)} panels from mask")
for i, v in enumerate(panels_raw):
    print(f"  panel {i}: {len(v)} verts")

# ---- Edge snapping ----
# Three independent gutter signals, each computed as a grayscale image we'll
# evaluate for "peakiness" at every scan position:
#   (1) brightness — bright strips (classic white gutters)
#   (2) darkness   — dark ink borders between bright panels (inverted polarity)
#   (3) gradient   — tonal transitions with no visible strip (gutter implied
#                    only by a content change between adjacent panels)
# At each scan position we compute a peakiness score for each signal — center
# minus mean of both sides a few px away — and take the best. Flat bright
# regions (speech bubbles, pizza boxes) score near zero on all three.
gray_f = orig_gray.astype(np.float32)
dark_f = 255.0 - gray_f
gx = cv2.Sobel(orig_gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(orig_gray, cv2.CV_32F, 0, 1, ksize=3)
grad_f = cv2.magnitude(gx, gy)
grad_f = cv2.normalize(grad_f, None, 0, 255, cv2.NORM_MINMAX)

# Scale parameters to image dimensions so the script works across resolutions.
PAGE_DIAG = float(np.hypot(W, H))
EDGE_MARGIN    = max(4, int(PAGE_DIAG * 0.004))    # ~7 px at 1024x1536
SCAN_HALF      = max(6, int(PAGE_DIAG * 0.008))    # ~14 px (symmetric fallback)
PEAK_CONTEXT   = max(6, int(PAGE_DIAG * 0.007))    # ~12 px
N_SAMPLES      = 40
MIN_PEAK       = 25        # minimum peak strength to accept as boundary signal
MIN_SNAPPED    = 10

# Cover-line strategy: the mask should hide all of a panel's content
# (including bleed that juts into adjacent panels). Scan deep OUTWARD from
# the panel centroid to reach bleed boundaries; scan shallow INWARD to
# allow small corrections where the mask sits slightly too far out.
INWARD_MAX     = max(6, int(PAGE_DIAG * 0.006))    # ~11 px
OUTWARD_MAX    = max(40, int(PAGE_DIAG * 0.08))    # ~148 px at 1024x1536

def is_page_edge_point(x, y):
    return x <= EDGE_MARGIN or x >= W - EDGE_MARGIN or y <= EDGE_MARGIN or y >= H - EDGE_MARGIN

def is_page_edge(v1, v2):
    """An edge is a page-edge if BOTH endpoints lie on the same page boundary."""
    x1, y1 = v1; x2, y2 = v2
    if x1 <= EDGE_MARGIN and x2 <= EDGE_MARGIN: return True
    if x1 >= W - EDGE_MARGIN and x2 >= W - EDGE_MARGIN: return True
    if y1 <= EDGE_MARGIN and y2 <= EDGE_MARGIN: return True
    if y1 >= H - EDGE_MARGIN and y2 >= H - EDGE_MARGIN: return True
    return False

def snap_edge(v1, v2, centroid=None):
    """Return (m, b, horiz_flag) representing the snapped line equation.
    horiz_flag=True if parameterised as y=m*x+b (more horizontal), else x=m*y+b.

    If centroid is provided, bias snap outward from centroid — prefer snap
    positions that are slightly further from the panel center rather than
    inward. This honors reading-direction convention (mask covers slightly
    MORE of the adjacent panel, never less, so no unread content peeks out).
    """
    x1, y1 = v1; x2, y2 = v2
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    if L < 5: return None
    # Unit tangent and perpendicular
    tx, ty = dx / L, dy / L
    nx, ny = -ty, tx  # rotate 90° CCW

    # Determine which perpendicular direction points AWAY from centroid
    outward_sign = 0
    if centroid is not None:
        midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
        dot = nx * (midx - centroid[0]) + ny * (midy - centroid[1])
        outward_sign = 1 if dot > 0 else -1

    horiz = abs(dx) >= abs(dy)

    # Sample points along the edge
    samples = []
    for i in range(N_SAMPLES):
        t = (i + 0.5) / N_SAMPLES
        sx = x1 + t * dx
        sy = y1 + t * dy
        # Symmetric scan for the strongest gutter-like peak. Multi-polarity
        # scoring handles bright strips, dark lines, and tonal transitions.
        # Gentle outward bias nudges the fit toward the "safe" side (covering
        # slightly more of the adjacent read panel) when peakiness is tied.
        best_score = -1e9
        best_off = None
        for off in range(-SCAN_HALF, SCAN_HALF + 1):
            px = int(round(sx + off * nx))
            py = int(round(sy + off * ny))
            if not (0 <= px < W and 0 <= py < H):
                continue
            lx = int(round(sx + (off - PEAK_CONTEXT) * nx))
            ly = int(round(sy + (off - PEAK_CONTEXT) * ny))
            rx = int(round(sx + (off + PEAK_CONTEXT) * nx))
            ry = int(round(sy + (off + PEAK_CONTEXT) * ny))
            if not (0 <= lx < W and 0 <= ly < H and 0 <= rx < W and 0 <= ry < H):
                continue
            b_c, b_l, b_r = gray_f[py, px], gray_f[ly, lx], gray_f[ry, rx]
            d_c, d_l, d_r = dark_f[py, px], dark_f[ly, lx], dark_f[ry, rx]
            g_c, g_l, g_r = grad_f[py, px], grad_f[ly, lx], grad_f[ry, rx]
            s_bright = b_c - max(b_l, b_r)
            s_dark   = d_c - max(d_l, d_r)
            s_grad   = g_c - max(g_l, g_r)
            OUTWARD_BIAS = 2.5
            score = max(s_bright, s_dark, s_grad) + outward_sign * off * OUTWARD_BIAS
            if score > best_score:
                best_score = score
                best_off = off

        if best_off is not None and best_score >= MIN_PEAK:
            samples.append((sx + best_off * nx, sy + best_off * ny))

    if len(samples) < MIN_SNAPPED:
        return None

    xs = np.array([p[0] for p in samples], float)
    ys = np.array([p[1] for p in samples], float)

    # Robust line fit (iterative 2σ rejection)
    if horiz:
        for _ in range(4):
            mk, bk = np.polyfit(xs, ys, 1)
            r = ys - (mk * xs + bk)
            sd = r.std()
            if sd < 0.5: break
            keep = np.abs(r) < 2 * sd
            if keep.sum() < MIN_SNAPPED or keep.all(): break
            xs, ys = xs[keep], ys[keep]
        mk, bk = np.polyfit(xs, ys, 1)
        return (float(mk), float(bk), True)
    else:
        for _ in range(4):
            mk, bk = np.polyfit(ys, xs, 1)
            r = xs - (mk * ys + bk)
            sd = r.std()
            if sd < 0.5: break
            keep = np.abs(r) < 2 * sd
            if keep.sum() < MIN_SNAPPED or keep.all(): break
            xs, ys = xs[keep], ys[keep]
        mk, bk = np.polyfit(ys, xs, 1)
        return (float(mk), float(bk), False)

def line_to_page_line(v1, v2, fit):
    """If edge is a page-edge (both vertices on same page boundary), return
    a synthetic line equation for that boundary. Otherwise use fit."""
    if fit is not None:
        return fit
    # Fallback: keep the original line through v1, v2 as a fit
    x1, y1 = v1; x2, y2 = v2
    horiz = abs(x2 - x1) >= abs(y2 - y1)
    if horiz:
        if x2 == x1: return (0.0, float(y1), True)
        mk = (y2 - y1) / (x2 - x1)
        bk = y1 - mk * x1
        return (float(mk), float(bk), True)
    else:
        if y2 == y1: return (0.0, float(x1), False)
        mk = (x2 - x1) / (y2 - y1)
        bk = x1 - mk * y1
        return (float(mk), float(bk), False)

def intersect_lines(L1, L2):
    m1, b1, h1 = L1
    m2, b2, h2 = L2
    # Four cases depending on parameterisations
    if h1 and h2:
        # y = m1*x + b1 and y = m2*x + b2
        if abs(m1 - m2) < 1e-6: return None
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    elif not h1 and not h2:
        if abs(m1 - m2) < 1e-6: return None
        y = (b2 - b1) / (m1 - m2)
        x = m1 * y + b1
    elif h1 and not h2:
        # y = m1*x + b1  and  x = m2*y + b2
        denom = 1 - m1 * m2
        if abs(denom) < 1e-6: return None
        y = (m1 * b2 + b1) / denom
        x = m2 * y + b2
    else:
        # x = m1*y + b1  and  y = m2*x + b2
        denom = 1 - m1 * m2
        if abs(denom) < 1e-6: return None
        x = (m1 * b2 + b1) / denom
        y = m2 * x + b2
    return (x, y)

# ---- Snap each panel's edges ----
panels_edges = []    # list of (centroid, edge_list) per panel; edge = (kind, line, v1, v2)
for verts in panels_raw:
    n = len(verts)
    cx = sum(v[0] for v in verts) / n
    cy = sum(v[1] for v in verts) / n
    centroid = (cx, cy)
    edge_lines = []
    for i in range(n):
        v1 = verts[i]
        v2 = verts[(i + 1) % n]
        if is_page_edge(v1, v2):
            line = line_to_page_line(v1, v2, None)
            edge_lines.append(("page", line, v1, v2))
            continue
        fit = snap_edge(v1, v2, centroid)
        if fit is None:
            fit = line_to_page_line(v1, v2, None)
            edge_lines.append(("orig", fit, v1, v2))
        else:
            edge_lines.append(("snap", fit, v1, v2))
    panels_edges.append((centroid, edge_lines, verts))

# ---- Shared-gutter merging: find pairs of edges between adjacent panels ----
# that share a gutter and average their snapped lines, so both panels agree
# on one boundary. Without this, outward bias pushes each side in opposite
# directions and they diverge, leaving a misaligned strip at the shared edge.
def edge_midpoint(v1, v2):
    return ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2)

def edge_length(v1, v2):
    return np.hypot(v2[0] - v1[0], v2[1] - v1[1])

def edge_direction(v1, v2):
    L = edge_length(v1, v2)
    if L < 1: return (0, 0)
    return ((v2[0] - v1[0]) / L, (v2[1] - v1[1]) / L)

def are_shared(e_i, e_j, max_dist=40, max_angle_dev=np.radians(15)):
    """Two snapped edges share a gutter if their midpoints are close AND
    their directions are parallel (or anti-parallel)."""
    _, _, v_i1, v_i2 = e_i
    _, _, v_j1, v_j2 = e_j
    m_i = edge_midpoint(v_i1, v_i2)
    m_j = edge_midpoint(v_j1, v_j2)
    d = np.hypot(m_i[0] - m_j[0], m_i[1] - m_j[1])
    if d > max_dist:
        return False
    d_i = edge_direction(v_i1, v_i2)
    d_j = edge_direction(v_j1, v_j2)
    dot = abs(d_i[0] * d_j[0] + d_i[1] * d_j[1])  # 1=parallel, 0=perpendicular
    return dot > np.cos(max_angle_dev)

def average_lines(L1, L2):
    """Average two line equations (same parameterisation)."""
    m1, b1, h1 = L1
    m2, b2, h2 = L2
    if h1 == h2:
        return ((m1 + m2) / 2, (b1 + b2) / 2, h1)
    return L1

def wide_snap_shared_gutter(v1, v2, max_offset=100):
    """Wide-search snap specifically for shared gutters. Tests a wide range of
    perpendicular offsets and picks the offset with highest AVERAGED peakiness
    across the whole edge. This filters out local noise (character contours,
    text edges): they score high at a few samples but not consistently. A real
    gutter scores high consistently along its full length.

    Returns a line equation (m, b, horiz_flag) shifted to the best offset,
    preserving the input edge's slope."""
    x1, y1 = v1; x2, y2 = v2
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    if L < 5: return None
    tx, ty = dx / L, dy / L
    nx, ny = -ty, tx

    # Precompute sample anchor positions along the edge
    anchors = []
    for i in range(N_SAMPLES):
        t = (i + 0.5) / N_SAMPLES
        anchors.append((x1 + t * dx, y1 + t * dy))

    def peakiness_at(px, py, lx, ly, rx, ry):
        b_c, b_l, b_r = gray_f[py, px], gray_f[ly, lx], gray_f[ry, rx]
        d_c, d_l, d_r = dark_f[py, px], dark_f[ly, lx], dark_f[ry, rx]
        g_c, g_l, g_r = grad_f[py, px], grad_f[ly, lx], grad_f[ry, rx]
        return max(b_c - max(b_l, b_r),
                   d_c - max(d_l, d_r),
                   g_c - max(g_l, g_r))

    best_off = 0
    best_avg = -1e9
    # Step of 2 is plenty for finding the right neighborhood; we refine below.
    for off in range(-max_offset, max_offset + 1, 2):
        scores = []
        for (sx, sy) in anchors:
            px = int(round(sx + off * nx))
            py = int(round(sy + off * ny))
            if not (0 <= px < W and 0 <= py < H): continue
            lx = int(round(sx + (off - PEAK_CONTEXT) * nx))
            ly = int(round(sy + (off - PEAK_CONTEXT) * ny))
            rx = int(round(sx + (off + PEAK_CONTEXT) * nx))
            ry = int(round(sy + (off + PEAK_CONTEXT) * ny))
            if not (0 <= lx < W and 0 <= ly < H and 0 <= rx < W and 0 <= ry < H):
                continue
            scores.append(peakiness_at(px, py, lx, ly, rx, ry))
        if len(scores) < N_SAMPLES * 0.5:
            continue
        avg = sum(scores) / len(scores)
        if avg > best_avg:
            best_avg = avg
            best_off = off

    if best_avg < MIN_PEAK * 0.5:  # still no coherent signal anywhere
        return None

    # Compute a line equation for the input edge SHIFTED by best_off along
    # its perpendicular (nx, ny). The shifted line passes through
    # (x1 + best_off*nx, y1 + best_off*ny) with the original slope.
    sx1 = x1 + best_off * nx
    sy1 = y1 + best_off * ny
    sx2 = x2 + best_off * nx
    sy2 = y2 + best_off * ny
    horiz = abs(dx) >= abs(dy)
    if horiz:
        if abs(sx2 - sx1) < 1e-6:
            return (0.0, sy1, True)
        m = (sy2 - sy1) / (sx2 - sx1)
        b = sy1 - m * sx1
        return (float(m), float(b), True)
    else:
        if abs(sy2 - sy1) < 1e-6:
            return (0.0, sx1, False)
        m = (sx2 - sx1) / (sy2 - sy1)
        b = sx1 - m * sy1
        return (float(m), float(b), False)

merges = 0
for i in range(len(panels_edges)):
    for j in range(i + 1, len(panels_edges)):
        _, edges_i, _ = panels_edges[i]
        _, edges_j, _ = panels_edges[j]
        for ei_idx, e_i in enumerate(edges_i):
            if e_i[0] in ("page", "orig"): continue
            for ej_idx, e_j in enumerate(edges_j):
                if e_j[0] in ("page", "orig"): continue
                if are_shared(e_i, e_j):
                    # Re-snap jointly with wide search + averaged peakiness.
                    # This can correct mask errors much larger than the narrow
                    # snap's ±SCAN_HALF window.
                    new_line = wide_snap_shared_gutter(e_i[2], e_i[3])
                    if new_line is None:
                        # Fallback: average the already-snapped lines
                        new_line = average_lines(e_i[1], e_j[1])
                    edges_i[ei_idx] = ("shared", new_line, e_i[2], e_i[3])
                    edges_j[ej_idx] = ("shared", new_line, e_j[2], e_j[3])
                    merges += 1
print(f"Merged {merges} shared gutter(s)")

# Now compute final polygons using the (possibly merged) edges
panels_refined = []
for centroid, edge_lines, verts in panels_edges:
    n = len(verts)
    kinds = [e[0] for e in edge_lines]
    print(f"  refined panel: {kinds}")

    # New vertices = intersection of edge i-1 and edge i
    new_verts = []
    for i in range(n):
        _, L_prev, _, _ = edge_lines[(i - 1) % n]
        _, L_curr, _, _ = edge_lines[i]
        p = intersect_lines(L_prev, L_curr)
        if p is None:
            p = verts[i]
        x, y = p
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        new_verts.append((int(round(x)), int(round(y))))
    panels_refined.append(new_verts)

# ---- Visualize: original mask boundary in magenta, refined in cyan ----
viz = orig_bgr.copy()
# Draw original mask boundary
edges_old = cv2.Canny(m, 50, 150)
edges_old = cv2.dilate(edges_old, np.ones((3, 3), np.uint8))
viz[edges_old > 0] = (255, 0, 255)      # magenta
# Draw refined polygons in cyan
for verts in panels_refined:
    pts = np.array(verts, np.int32).reshape(-1, 1, 2)
    cv2.polylines(viz, [pts], isClosed=True, color=(255, 255, 0), thickness=3)

cv2.imwrite(f"{page.stem}_snapped_compare.png", viz)
print(f"\nSaved {page.stem}_snapped_compare.png  (magenta=mask, cyan=snapped)")

# Build the refined mask image
refined_mask = np.zeros((H, W), np.uint8)
for verts in panels_refined:
    pts = np.array(verts, np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(refined_mask, [pts], 255)
cv2.imwrite(f"{page.stem}_snapped_mask.png", refined_mask)
print(f"Saved {page.stem}_snapped_mask.png")

# Visualize the refined mask overlaid on the original
tint = orig_bgr.copy().astype(np.float32)
colors = [(0,0,230), (230,140,0), (0,200,230), (0,230,120), (200,0,230), (230,140,0)]
pil_orig = Image.open(page).convert("RGBA")
overlay = Image.new("RGBA", pil_orig.size, (0,0,0,0))
draw = ImageDraw.Draw(overlay)
for i, verts in enumerate(panels_refined):
    c = colors[i % len(colors)]
    draw.polygon(verts, fill=(*c, 110), outline=(255,255,255,255))
Image.alpha_composite(pil_orig, overlay).convert("RGB").save(f"{page.stem}_snapped_overlay.png")
print(f"Saved {page.stem}_snapped_overlay.png")

with open(f"{page.stem}_snapped_panels.json", "w") as f:
    json.dump(panels_refined, f, indent=2)
