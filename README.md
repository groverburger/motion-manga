<p align="center">
  <img src="logo.svg" width="380" alt="motion-manga">
</p>

A pipeline I built for turning manga pages into vertical motion-comic
videos. The output is 1080×1920 and runs about 16 seconds, sized for
TikTok / Reels / Shorts. There's camera movement between panels,
voiceover, sound effects, and small per-character motion.

## Demo

![pizza-blitz preview, silent, 2.5x speed](examples/pizza-blitz/final.gif)

That clip is silent and sped up. The full version with audio is at
[examples/pizza-blitz/final.mp4](examples/pizza-blitz/final.mp4).

## How it works

You drop one or more manga pages into a project directory, then point
a coding agent (Claude Code, etc.) at `SKILL.md` and let it drive.
Across five rough stages, the agent:

1. Asks GPT Image 2 for a binary panel mask, then snaps the rough
   polygon edges to real gutters using `tools/snap_mask.py`.
2. Asks GPT Image 2 again for a foreground mask, blurs the original
   page where the characters used to be, and splits the foreground
   into per-character RGBA layers via OpenCV connected components.
3. Generates voiceover and sound effects with ElevenLabs v3.
4. Fills in a Hyperframes composition starting from `template/`.
5. Renders the result to MP4.

Each stage is either a prompt or a small Python script. Any of them
can be swapped out.

## Quickstart

To render the included example:

```bash
cd examples/pizza-blitz
npx hyperframes render
```

To start a new project:

```bash
cp -r template my-manga
cp <your_page>.png my-manga/page1.png
export OPENAI_API_KEY=sk-...
export ELEVENLABS_API_KEY=...
```

Then ask your coding agent to follow `SKILL.md`.

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
- Node + npm, for `npx hyperframes`
- An OpenAI API key, for GPT Image 2 mask generation
- An ElevenLabs API key, for voice and sound effects

The Python scripts in `tools/` read `OPENAI_API_KEY` and
`ELEVENLABS_API_KEY` from the environment. Either `export` them once
per shell session or prefix individual calls.

## A note on scope

This doesn't generate page art. Bring your own pages, however you got
them. If you want pages from a prompt too, do that as a separate step
and feed the result in here.
