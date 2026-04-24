"""Panel detection via vision LLM. Zero local model — delegates the spatial
reasoning to a multimodal model that actually understands manga layout."""
import base64, json, pathlib, sys
from openai import OpenAI

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "page1.png")
stem = src.stem

client = OpenAI(api_key=pathlib.Path.home().joinpath("Documents/openai_api_key.txt").read_text().strip())

b64 = base64.b64encode(src.read_bytes()).decode()

prompt = """You are analyzing a manga page. The image is exactly 1024x1536 pixels.

Identify every comic panel on this page, in reading order (top-to-bottom, left-to-right).
For each panel, return the polygon of its ACTUAL shape — tracing the real gutter boundaries.
Panels can have angled gutters (trapezoidal shapes) and can be CONCAVE (e.g. when a character
figure breaks through the gutter border between two panels).

Important rules:
- Coordinates are in image pixels with origin at top-left.
- A panel's vertices should trace its outer boundary in clockwise order.
- Most panels will have 4–6 vertices. A panel with a character breaking the frame
  may have 6–10 vertices tracing the notch.
- Gutter lines should follow where the real gutter sits on the page — not rough rectangles.
- If a panel bleeds to the page edge, include (0, …) or (1024, …) / (…, 1536) vertices.

Return ONLY valid JSON of the form:
{"panels": [{"name": "p1", "vertices": [[x1,y1], [x2,y2], ...]}, ...]}
No prose, no markdown fences."""

resp = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }],
    response_format={"type": "json_object"},
)

out = resp.choices[0].message.content
print(out[:500], "...")
data = json.loads(out)

# Save
with open(f"{stem}_panels_llm.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"\n{len(data['panels'])} panels saved to {stem}_panels_llm.json")

# Visualize
from PIL import Image, ImageDraw
viz = Image.open(src).convert("RGBA")
overlay = Image.new("RGBA", viz.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)
colors = [(255,0,0,110),(0,200,255,110),(255,220,0,110),(0,255,120,110),
          (200,0,255,110),(255,120,0,110),(0,100,255,110)]
for i, p in enumerate(data["panels"]):
    pts = [tuple(v) for v in p["vertices"]]
    draw.polygon(pts, fill=colors[i % len(colors)], outline=(255,255,255,255))
Image.alpha_composite(viz, overlay).convert("RGB").save(f"{stem}_polygons_llm.png")
print(f"Saved {stem}_polygons_llm.png")
