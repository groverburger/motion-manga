"""Overlay the generated mask's panel boundaries on the original page
so we can see exactly where the mask deviates from real panel edges."""
import cv2, sys, pathlib
import numpy as np
from PIL import Image

page = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "page1.png")
mask = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "page1_mask.png")

orig = cv2.imread(str(page))                            # BGR
m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
if m.shape != orig.shape[:2]:
    m = cv2.resize(m, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
_, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

# 1) Side-by-side: original | mask (scaled to match height)
h = orig.shape[0]
w = orig.shape[1]
mbgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
side = np.hstack([orig, np.full((h, 20, 3), 128, np.uint8), mbgr])

# 2) Mask-edge traced over original in magenta
edges = cv2.Canny(m, 50, 150)
edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edge_overlay = orig.copy()
edge_overlay[edges > 0] = (255, 0, 255)  # magenta

# 3) Tint: mask's white region as cyan wash over original; black region (gutters) as red wash.
# Makes panel coverage easy to eyeball against the page.
tint = orig.copy().astype(np.float32)
white_area = (m == 255)
black_area = (m == 0)
tint[white_area] = tint[white_area] * 0.65 + np.array([200, 240, 0], np.float32) * 0.35
tint[black_area] = tint[black_area] * 0.60 + np.array([0, 0, 240], np.float32) * 0.40
tint = tint.astype(np.uint8)

# Stack all three
row = np.hstack([orig, np.full((h, 12, 3), 64, np.uint8),
                 edge_overlay, np.full((h, 12, 3), 64, np.uint8),
                 tint])
cv2.imwrite(f"{page.stem}_overlay.png", row)

# Also write individual diagnostics
cv2.imwrite(f"{page.stem}_edges.png", edge_overlay)
cv2.imwrite(f"{page.stem}_tint.png", tint)
print(f"Saved {page.stem}_overlay.png (original | edges | tint)")
print(f"Saved {page.stem}_edges.png (mask boundary in magenta)")
print(f"Saved {page.stem}_tint.png (mask regions tinted)")
