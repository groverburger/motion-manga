"""Overlay proposed panel polygons on the page to verify coverage."""
from PIL import Image, ImageDraw

img = Image.open("page1.png").convert("RGBA")
overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

# Polygon estimates (x, y) — expand slightly past gutters so edges never leave gaps
# Each polygon follows the panel shape (trapezoid / quad).
panels = {
    "p1": [(0, 0),     (1024, 0),     (1024, 306),  (0, 260)],
    "p2": [(0, 314),   (506, 282),    (548, 704),   (0, 704)],
    "p3": [(514, 282), (1024, 310),   (1024, 704),  (556, 704)],
    "p4": [(0, 724),   (1024, 740),   (1024, 906),  (0, 926)],
    "p5": [(0, 930),   (1024, 914),   (1024, 1536), (0, 1536)],
}

colors = {
    "p1": (255, 0,   0,   110),
    "p2": (0,   200, 255, 110),
    "p3": (255, 220, 0,   110),
    "p4": (0,   255, 120, 110),
    "p5": (200, 0,   255, 110),
}

for name, pts in panels.items():
    draw.polygon(pts, fill=colors[name], outline=(255, 255, 255, 220))

out = Image.alpha_composite(img, overlay)
out.convert("RGB").save("panels_polygons.png")
print("Saved panels_polygons.png")
