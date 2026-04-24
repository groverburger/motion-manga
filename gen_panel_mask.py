"""Generate a binary panel-fills mask from a manga page using GPT Image 2.

Usage:  python3 gen_panel_mask.py <page.png>  [output_mask.png]

Output: a 1024x1536 near-binary PNG where every comic panel interior is white
(255) and all gutters, page borders, and non-panel regions are black (0).
Extract panel polygons with cv2.findContours on the thresholded result.
"""
import base64, pathlib, sys
from openai import OpenAI

PROMPT = """Output a BINARY DIAGRAM the same size as the input image.

Task: fill EVERY comic panel's interior with PURE WHITE (#FFFFFF), and fill
ONLY the gutters between panels with PURE BLACK (#000000).

Count every panel, including:
- Panels that bleed to the edge of the page (they still count as panels — fill
  them white all the way to the page boundary).
- Dark-toned panels (night scenes, black backgrounds) — they are still panels,
  just dark ones. Fill them solid white in the output.
- Small panels, reaction panels, splash panels — all panels.

A panel is any region separated from adjacent panels by a gutter (a thin strip
between panels). Start by counting the panels in the input and make sure every
one of them appears as a white region in your output.

Output constraints:
- Strictly two colors: pure white (#FFFFFF) for panel interiors, pure black
  (#000000) everywhere else. No grayscale. No manga artwork. No characters.
- The white/black boundary must trace each panel's actual gutter edges,
  including angled edges and any notches where a character figure breaks the
  panel frame.
- Same resolution as the input.
"""

def generate(src: pathlib.Path, out: pathlib.Path) -> None:
    client = OpenAI(
        api_key=pathlib.Path.home().joinpath("Documents/openai_api_key.txt").read_text().strip()
    )
    with open(src, "rb") as f:
        resp = client.images.edit(
            model="gpt-image-2",
            image=f,
            prompt=PROMPT,
            size="1024x1536",
            quality="high",
            n=1,
        )
    b64 = resp.data[0].b64_json
    if b64:
        out.write_bytes(base64.b64decode(b64))
    else:
        import urllib.request
        urllib.request.urlretrieve(resp.data[0].url, out)
    print(f"Saved {out}")

if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(f"{src.stem}_mask.png")
    generate(src, out)
