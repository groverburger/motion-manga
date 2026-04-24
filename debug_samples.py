"""Visualize all gutter-C samples — each candidate y at each x — so I can see
which samples are correct and which are false positives."""
from PIL import Image, ImageDraw
import numpy as np

img = Image.open("page1.png").convert("L")
W, H = img.size
arr = np.array(img)

y_lo, y_hi = 660, 770
min_brightness = 230
context = 18
dark_ceiling = 140

viz = img.convert("RGB")
draw = ImageDraw.Draw(viz)

samples = []
rejected = []  # reason, x, y
for x in range(0, W, 6):
    col = arr[:, x]
    band = col[y_lo:y_hi]
    order = np.argsort(band)[::-1]
    found = False
    for idx in order[:8]:
        if band[idx] < min_brightness:
            break
        y = y_lo + int(idx)
        above_lo = max(0, y - context - 12)
        above_hi = max(0, y - context)
        below_lo = min(H, y + context)
        below_hi = min(H, y + context + 12)
        if above_hi <= above_lo or below_hi <= below_lo:
            continue
        above_mean = col[above_lo:above_hi].mean()
        below_mean = col[below_lo:below_hi].mean()
        if above_mean < dark_ceiling and below_mean < dark_ceiling:
            samples.append((x, y))
            # green dot = accepted
            draw.ellipse([x-3, y-3, x+3, y+3], fill="#00ff00", outline="#000")
            found = True
            break
        else:
            rejected.append(("context", x, y, above_mean, below_mean))
    if not found:
        # look at brightest candidate to see where it was
        y_pk = y_lo + int(np.argmax(band))
        val = int(band[int(np.argmax(band))])
        if val >= 200:
            # red X = bright spot rejected or none found
            draw.ellipse([x-3, y_pk-3, x+3, y_pk+3], outline="#ff0000", width=2)

viz.save("gutter_c_debug.png")

# Also compute the fit and draw it
if len(samples) >= 4:
    xs = np.array([s[0] for s in samples], dtype=float)
    ys = np.array([s[1] for s in samples], dtype=float)
    # Look at the distribution of y values across x buckets
    print(f"Total accepted samples: {len(samples)}")
    print("\nSample y-values in x-buckets:")
    for lo in range(0, W, 200):
        hi = lo + 200
        bucket = [s[1] for s in samples if lo <= s[0] < hi]
        if bucket:
            print(f"  x={lo:4d}..{hi:4d}:  n={len(bucket):2d}  "
                  f"y_min={min(bucket)}  y_median={int(np.median(bucket))}  y_max={max(bucket)}")
        else:
            print(f"  x={lo:4d}..{hi:4d}:  no samples")

    # Robust fit
    for _ in range(5):
        m, b = np.polyfit(xs, ys, 1)
        resid = ys - (m * xs + b)
        sd = resid.std()
        if sd < 1e-6:
            break
        keep = np.abs(resid) < 2 * sd
        if keep.sum() < 4 or keep.all():
            break
        xs, ys = xs[keep], ys[keep]
    m, b = np.polyfit(xs, ys, 1)
    print(f"\nFinal fit: y = {m:+.4f}*x + {b:.2f}")
    print(f"  y(0)={b:.1f}  y(1023)={m*1023+b:.1f}")
    # Draw the fit line
    y0 = b; y1 = m*1023 + b
    draw.line([(0, y0), (1023, y1)], fill="#ff00ff", width=3)
    viz.save("gutter_c_debug.png")

print("\nSaved gutter_c_debug.png — green dots = accepted samples, "
      "red circles = bright but rejected, magenta = line fit")
