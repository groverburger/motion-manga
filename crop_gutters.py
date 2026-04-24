"""Crop each gutter region at full resolution for precise inspection."""
from PIL import Image

img = Image.open("page1.png")
W, H = img.size
print(f"Page: {W}x{H}")

# Region: top gutter (between panel 1 and panels 2/3)
img.crop((0, 220, W, 340)).save("gutter_top.png")
# Region: diagonal gutter (between panel 2 and panel 3)
img.crop((400, 260, 620, 720)).save("gutter_diag.png")
# Region: mid gutter (below panels 2/3, above panel 4)
img.crop((0, 670, W, 770)).save("gutter_mid.png")
# Region: lower gutter (between panel 4 and panel 5)
img.crop((0, 870, W, 970)).save("gutter_lower.png")
print("Saved 4 gutter crops")
