"""Cheap inpainting: blur the character region of the page image so parallax
motion doesn't expose a sharp doubled silhouette behind the fg layer.

Usage: python3 gen_blurred_bg.py <page.png> <fg_mask.png> [out.png]

Output: a PNG where pixels inside the fg mask are aggressively Gaussian
blurred (character dissolves into mush) and pixels outside are kept sharp.
When the fg layer drifts on top, any uncovered mush reads as "ghost of
character" rather than "sharp doubled silhouette."
"""
import pathlib, sys
from PIL import Image, ImageFilter

src_page  = pathlib.Path(sys.argv[1])
src_mask  = pathlib.Path(sys.argv[2])
out_path  = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else src_page.with_name(f"{src_page.stem}_bg.png")

BLUR_RADIUS = 40   # aggressive — kills character definition entirely
MASK_DILATE = 8    # expand mask a bit so the character's sharp outline
                   # is also blurred, not just the interior

page = Image.open(src_page).convert("RGB")
mask = Image.open(src_mask).convert("L")
if mask.size != page.size:
    mask = mask.resize(page.size, Image.NEAREST)

# Dilate the mask by blurring it and re-thresholding — grows the region
# slightly so the edges of the character are also blurred.
if MASK_DILATE > 0:
    mask = mask.filter(ImageFilter.GaussianBlur(radius=MASK_DILATE))

# Blur the entire page
blurred = page.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

# Composite: mask white → blurred, mask black → original sharp
# PIL.Image.composite(fg, bg, mask): mask=255 picks fg, mask=0 picks bg
result = Image.composite(blurred, page, mask)
result.save(out_path)
print(f"Saved {out_path}")
