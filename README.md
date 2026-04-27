# manga-motion

A factory for turning a manga page into a vertical motion-comic video.
Bring a generated or scanned page; the agent + scripts do the rest:
panel detection, character cutouts, voiceover, sound design, animation.

The output is a 16-second-ish 1080×1920 MP4 with camera movement,
panel-by-panel reveal, voiceover, sound effects, and subtle character
motion. Tuned for vertical short-form (TikTok / Reels / Shorts).

## How to use it

Drop your manga page(s) into a new project directory and ask a coding
agent (Claude Code, etc.) to follow `SKILL.md`. The agent will:

1. Generate a panel mask from the page (GPT Image 2) and snap the
   polygon edges to real gutters with `tools/snap_mask.py`.
2. Generate a foreground mask, blur the page where characters were,
   and split the foreground into per-character RGBA layers.
3. Generate voiceover and sound effects (ElevenLabs v3).
4. Fill in a Hyperframes composition based on `template/`.
5. Render to MP4.

For a worked example end-to-end, see `examples/pizza-blitz/`.

## Repo layout

```
manga-motion/
├── README.md            # this file
├── SKILL.md             # agent-facing skill: how to animate a page
├── PIPELINE.md          # detailed reference for each stage
├── tools/               # Python scripts for masks, cutouts, voice
├── template/            # boilerplate Hyperframes project to copy
└── examples/
    └── pizza-blitz/     # complete worked example (3 pages, 16s video)
```

## Prerequisites

- Python 3.10+ with `opencv-python`, `Pillow`, `numpy`, `shapely`,
  `openai`, `elevenlabs`
- Node + npm (for `npx hyperframes` — render pipeline)
- An OpenAI API key (panel + foreground masks via GPT Image 2)
- An ElevenLabs API key (voice + sound effects)

API keys are read from `~/Documents/openai_api_key.txt` and
`~/Documents/elevenlabs_api_key.txt` by the scripts under `tools/`.
Adjust the paths inside the scripts if yours live somewhere else.

## What this is not

Not a page-art generator. The pipeline assumes your pages already
exist — drawn, scanned, or generated elsewhere. If you also want to
generate pages from a prompt, do that step separately and bring the
result here.
