<p align="center">
  <img src="logo.svg" width="380" alt="motion-manga">
</p>

A factory for turning a manga page into a vertical motion-comic video.
Bring a generated or scanned page; the agent + scripts do the rest:
panel detection, character cutouts, voiceover, sound design, animation.

The output is a 16-second-ish 1080×1920 MP4 with camera movement,
panel-by-panel reveal, voiceover, sound effects, and subtle character
motion. Tuned for vertical short-form (TikTok / Reels / Shorts).

## What it produces

![pizza-blitz preview (silent, 2.5x speed)](examples/pizza-blitz/final.gif)

The clip above is a sped-up, silent preview of `examples/pizza-blitz/`
— the full worked example, with audio, is in
[`examples/pizza-blitz/final.mp4`](examples/pizza-blitz/final.mp4).

## How it works

Drop your manga page(s) into a new project directory and ask a coding
agent (Claude Code, etc.) to follow `SKILL.md`. The agent will:

1. Generate a panel mask from the page (GPT Image 2) and snap the
   polygon edges to real gutters with `tools/snap_mask.py`.
2. Generate a foreground mask, blur the page where characters were,
   and split the foreground into per-character RGBA layers.
3. Generate voiceover and sound effects (ElevenLabs v3).
4. Fill in a Hyperframes composition based on `template/`.
5. Render to MP4.

## Quickstart

Render the example:

```bash
cd examples/pizza-blitz
npx hyperframes render        # → renders/pizza-blitz_<timestamp>.mp4
```

Start a new project (the agent will fill in the gaps as it walks
through `SKILL.md`):

```bash
cp -r template my-manga
cp <your_page>.png my-manga/page1.png
export OPENAI_API_KEY=sk-...
export ELEVENLABS_API_KEY=...
# then point your coding agent at SKILL.md and let it drive
```

## Repo layout

```
motion-manga/
├── README.md            # this file
├── SKILL.md             # agent-facing skill: how to animate a page
├── PIPELINE.md          # detailed reference for each stage
├── logo.svg
├── tools/               # Python scripts for masks + cutouts
├── template/            # boilerplate Hyperframes project to copy
└── examples/
    └── pizza-blitz/     # complete worked example (3 pages, 16 s video)
```

## Prerequisites

- Python 3.10+ with `opencv-python`, `Pillow`, `numpy`, `shapely`,
  `openai`, `requests`
- Node + npm (for `npx hyperframes` — render pipeline)
- An OpenAI API key (panel + foreground masks via GPT Image 2)
- An ElevenLabs API key (voice + sound effects)

The scripts under `tools/` read `OPENAI_API_KEY` and `ELEVENLABS_API_KEY`
from the environment. Either `export` them once for your shell session
or prefix the call (`OPENAI_API_KEY=sk-... python3 tools/...`). The
agent operating this factory is expected to know the keys from
conversation, or to ask the user for them.

## What this is not

Not a page-art generator. The pipeline assumes your pages already
exist — drawn, scanned, or generated elsewhere. If you also want to
generate pages from a prompt, do that step separately and bring the
result here.
