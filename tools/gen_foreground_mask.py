"""Generate a binary foreground mask from a manga page using GPT Image 2.

Same pattern as gen_panel_mask_v2.py, different target: characters + held
objects in white, everything else (environment, borders, text, SFX) in black.

Usage:  python3 gen_foreground_mask.py <page.png>  [output_mask.png]
"""
import base64, pathlib, sys
from openai import OpenAI

PROMPT = """Output a BINARY DIAGRAM the same size as the input image.

Task: fill every CHARACTER and held-object silhouette with PURE WHITE
(#FFFFFF), and fill EVERYTHING ELSE with PURE BLACK (#000000).

Fill WHITE:
- Every character's body, face, hair, clothing
- Objects held by or attached to characters (pizza box, scooter being
  ridden, weapons, bags)
- Any pets or companions

Fill BLACK:
- Environment and scenery (buildings, walls, streets, sky, floor, rain,
  interior settings)
- Panel borders and gutters between panels
- Speech bubbles, thought bubbles, caption boxes, SFX lettering, any
  text of any kind
- Speed lines, motion lines, screentone shading when not on a character
- Anything a character is NOT actively holding

Output constraints:
- Strictly two colors — pure white (#FFFFFF), pure black (#000000). No
  grayscale, no manga linework, no detail rendering.
- The white/black boundary must trace the ACTUAL SILHOUETTE of each
  character — every hair spike, finger, coat flare, held-object edge.
- If there are multiple characters on the page (across panels), fill ALL
  of them white. A single character shown in multiple panels still gets
  filled white in each panel.
- Do not draw a black frame or margin around the page.
- Same resolution as the input.
"""

def generate(src: pathlib.Path, out: pathlib.Path) -> None:
    client = OpenAI()  # reads OPENAI_API_KEY from env
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
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(f"{src.stem}_fg_mask.png")
    generate(src, out)
