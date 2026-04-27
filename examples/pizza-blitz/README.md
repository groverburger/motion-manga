# pizza-blitz

A worked example of the manga-motion pipeline: a 16.5-second vertical
motion comic for the first page of a shonen-style parody about a pizza
delivery guy who treats his job with way too much gravity.

The animation exercises every stage of the factory:

- **Stage 2** — panel detection: GPT Image 2 binary mask refined by
  `tools/snap_mask.py` to produce 5 polygons (in `page1_v2_snapped_panels.json`)
- **Stage 2.5** — foreground cutouts: 7 connected components split out
  of `page1_fg_mask.png`, used for subtle per-character motion
  (`page1_fg_c{0..6}.png` + `page1_bg.png`)
- **Stage 3** — voice + SFX: 4 voiceover takes (Blitz + a confused
  customer) + 4 sound effects, in `audio/`
- **Stage 4** — animation: camera pans, push-ins, "dark until read"
  panel reveals, climactic punch frame on the final beat

## Render it

```bash
npx hyperframes lint     # should pass (one warning about mask3, ignorable)
npx hyperframes render   # MP4 lands under renders/ (gitignored)
```

About 30 seconds on a recent laptop. The killswitch pattern from
`SKILL.md §4h` is recommended in case the headless browser hangs.

The committed `final.mp4` in this directory is the canonical reference
output — what your render should look like if everything wired up
correctly.

## What this example demonstrates

If you're new to the pipeline, read `index.html` top to bottom — every
helper documented in `SKILL.md §4` is used here, and the per-component
motion patterns from `§4c.1` show up in the P3 hold (Blitz running,
shrinking 1% + drifting +8 px) and the P5 final hold (pizza box growing
1% + drifting +8 px).

The blurred background trick is most visible if you load `page1_bg.png`
on its own — the character regions are smeared into mush, but the
sharp `page1_fg_c{0..6}.png` cutouts on top reconstruct the page
seamlessly.
