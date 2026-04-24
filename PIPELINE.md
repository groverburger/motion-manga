# Manga → Motion-Comic Pipeline

End-to-end guide from story beats to a vertical-format animated video, as
worked out iteratively on the Pizza Blitz pages.

---

## 0. The four stages

```
┌──────────────┐  ┌────────────────────┐  ┌──────────────┐  ┌──────────────┐
│ 1. Page      │→ │ 2. Panel segment.  │→ │ 3. Audio     │→ │ 4. Animation │
│  (GPT Img 2) │  │    (mask + snap)   │  │ (ElevenLabs) │  │  (Hyperframes│
│              │  │                    │  │              │  │   + GSAP)    │
└──────────────┘  └────────────────────┘  └──────────────┘  └──────────────┘
  prompt + refs→    page.png →              .mp3 per line,    panels.json
  page.png         panels.json              per SFX           + audio clips
                                                              → page.mp4
```

Each stage is an independent script; no monolith. You can regenerate one
panel without touching the others, re-snap the masks without regenerating
the page, re-cut the audio without redoing panels, or re-time the animation
without re-recording anything.

---

## 1. Page generation (GPT Image 2)

### 1.1 Prompt structure that works

Treat the page like a director's shot list, panel by panel. Each beat:
who, what framing, what dialogue/SFX, what mood. 4–6 panels is a sweet
spot for 1024×1536 output.

```
Style block (top):
  - "authentic black-and-white shonen manga page"
  - paneling/screentone/crosshatch keywords
  - tone description ("absurdly serious tone applied to mundane subject,
    played completely straight")

Per panel:
  - framing (close-up / medium / wide / splash)
  - action
  - dialogue or caption text (in quotes, in English — will be rendered)
  - SFX (Japanese-style kanji OK, English onomatopoeia also works)

Style constraints (bottom):
  - "authentic shonen manga aesthetic, ink-and-tone only, no color"
  - "legible English dialogue and SFX"
  - "portrait orientation"
```

### 1.2 Moderation gotchas

- Naming specific copyrighted works ("Death Note") or living-artist styles
  ("by Takeshi Obata") gets rejected. Use genre descriptors instead:
  "sharp psychological-thriller shonen manga art style."
- Real character names from existing IP = rejected. Use your own names.

### 1.3 Character consistency across pages

Use `client.images.edit(image=canonical_page, prompt=...)` for every
subsequent page. Pass the FIRST page (or a dedicated character-sheet
image) as the reference. Don't pass the most recent page — character
drift compounds.

### 1.4 Output

`page{N}.png` at 1024×1536.

---

## 2. Panel segmentation

The image model understands panels semantically but is imprecise
geometrically. Classical CV is precise but fails on ambiguous gutters.
We combine both: the model tells us the topology (how many panels, how
they're arranged), CV refines the geometry (exactly where the lines are).

### 2.1 Mask generation — `gen_panel_mask_v2.py`

```bash
python3 gen_panel_mask_v2.py page1.png page1_mask.png
```

Asks GPT Image 2 to produce a binary panel-fills diagram: every panel
interior pure white, every gutter pure black, no black frame around the
page edges (panels that bleed to the edge extend white all the way).

**Key prompt requirements:**
- Strictly two colors (no grayscale, no manga artwork)
- Count every panel — including dark-toned panels and bleeding panels
- The v2 prompt explicitly forbids a page-edge frame (the v1 prompt drew
  one, leaving bleed panels short)

Output mask is ~90% binary with some anti-aliased edges (threshold during
extraction). Topology (panel count + adjacency) is nearly always correct.
Geometry is accurate to ±50–100 px.

### 2.2 Mask refinement — `snap_mask.py`

```bash
python3 snap_mask.py page1.png page1_mask.png
# produces page1_snapped_panels.json (polygon per panel)
#       + page1_snapped_overlay.png (visual verification)
```

Takes the rough mask and snaps its polygon edges to real gutter evidence
in the source page. The pipeline is:

1. **Extract panel polygons from mask** via `findContours` + `approxPolyDP`.
2. **Classify each polygon edge** as either `page-edge` (both endpoints
   sit against a page boundary) or `gutter-edge` (internal).
3. **For each gutter edge**, snap to real boundary signal:
   - Sample N positions along the edge.
   - At each position, scan perpendicular for the strongest "peak" score.
   - **Multi-polarity peakiness** catches three gutter types:
     - Bright strips (classic white gutters): `brightness - max(sides)`
     - Dark strips (ink-border panels): `darkness - max(sides)`
     - Step transitions (tonal change, no visible strip): `gradient -
       max(sides)`
   - Gentle **outward bias** (`+2.5/px`) nudges the fit to the "safe" side
     of ambiguity — the mask covers slightly more of the adjacent read
     panel rather than exposing the unread one.
   - **Robust line fit** (iterative 2σ outlier rejection) through the
     accepted samples.
4. **Detect shared gutters** between adjacent panels by checking edge-pair
   proximity + parallelism. For each shared pair:
   - Re-snap jointly using **averaged peakiness over the full edge length**
     with a **wide search window** (±100 px).
   - Averaging rejects local noise (character contours, text): real
     gutters score high at every sample; noise scores high at only a few.
   - Wide window handles cases where the mask was far off (100+ px).
   - Both panels' edges get the same resulting line — no more divergence
     from opposite outward biases cancelling out.
5. **Compute vertices** as intersections of adjacent edge-lines.

### 2.3 What the snap handles well

- Angled gutters (arbitrary slope)
- Diagonal gutters between side-by-side panels
- Faint gutters (multi-polarity peakiness)
- Dark-polarity gutters (ink borders between bright panels)
- Mask errors up to ~100 px (via shared-gutter wide snap)
- Page bleeds (preserved without snap)

### 2.4 What it doesn't yet handle

- **Concavity.** Each edge is fit as a single straight line. A panel with
  a character figure breaking the frame (notch) gets flattened. Principled
  fix would split edges at high-residual points and fit piecewise linear.
- **Closure validation.** Adjacent panels' corners aren't forced to agree
  exactly. Sub-pixel misalignments possible.
- **Topology errors from the mask-gen step.** If the image model splits
  one panel into two (or misses one), the snap can't recover. Mask-gen
  prompt iteration or N-of-M consensus masking would address this.

### 2.5 Output format

`page{N}_snapped_panels.json`:
```json
[
  [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],   // panel in detection order
  ...
]
```

Detection order is by area, **NOT reading order**. Remap to reading order
when consuming.

---

## 3. Motion-comic video (Hyperframes + GSAP)

The "dark until read" reveal effect: each panel starts hidden behind an
opaque black mask; masks fade as the camera visits each panel in reading
order.

### 3.1 Polygon preparation

Two layers of padding on the raw snapped polygons before use in the video:

**Layer 1: Shapely buffer (24 px outward).** Uniformly expands each
polygon. Needed because the 32 px CSS blur (see 3.3) fades the mask's
opacity near its hard edges — without buffer, the edge fade would bite
into the real panel content.

**Layer 2: Page-edge extension (48 px past page boundary).** For vertices
that sit on a page edge (x<0, x>W, y<0, y>H after buffer), push them
further — x to ±48 past W, y to ±48 past H. The 32 px blur radius means
the mask's full opacity zone is ~32 px inside its hard edge; without the
extension, the mask becomes translucent inside the page viewport at the
edges, letting unread-panel content leak through the bottom (or sides)
of the frame.

Script (inline in the doc, or easily extracted):
```python
from shapely.geometry import Polygon
W, H, BLEED = 1024, 1536, 48
p_buffered = Polygon(pts).buffer(24, join_style=2)
coords = list(p_buffered.exterior.coords)[:-1]
extended = []
for x, y in coords:
    if x < 0: x = -BLEED
    elif x >= W: x = W + BLEED
    if y < 0: y = -BLEED
    elif y >= H: y = H + BLEED
    extended.append((int(round(x)), int(round(y))))
```

### 3.2 SVG mask element

```html
<svg id="masks-svg" viewBox="-80 -80 1184 1696"
     preserveAspectRatio="none">
  <polygon id="mask1" points="..." />
  ...
</svg>
```

```css
#masks-svg {
  position: absolute;
  top: -80px;  left: -80px;       /* offset matches viewBox origin */
  width: 1184px; height: 1696px;  /* extended to fit +48 bleed +32 blur */
  pointer-events: none;
  filter: blur(32px);
}
#masks-svg polygon {
  fill: #000;
  will-change: opacity;
}
```

The SVG's viewBox and CSS dimensions must accommodate the extended
polygons PLUS blur room on all sides. Don't clip.

Each polygon fills pure black. GSAP animates `opacity` from 1 to 0 to
reveal the panel below.

### 3.3 Blur amount

32 px radius. Softens mask edges so any remaining snap imprecision fades
gracefully rather than showing as a sharp misaligned line.

This is why we buffer the polygons by 24 px: the blur eats ~32 px of hard
opacity at every edge. Without buffer, the mask's fully-opaque region sits
inside the real panel, letting content bleed. With 24 px buffer, the fully-
opaque region sits AT the real panel's boundary.

### 3.4 Animation camera — the `focus()` pattern

Root cause of early "drift up-and-to-the-left" during push-ins:
`transform-origin: 0 0` combined with `(x, y)` translates not recomputed
when scale changes. Fix:

```js
const PANELS = {
  p1: { cx: 512, cy: 154, s: 1.055 },
  p2: { cx: 267, cy: 519, s: 1.930 },
  // ...
};
function focus(p, extraScale = 1) {
  const s = p.s * extraScale;
  return {
    scale: s,
    x: VIEW_W/2 - s * p.cx,
    y: VIEW_H/2 - s * p.cy,
  };
}
```

Store each panel as `{centroid, base_scale}`. `focus()` computes transform
so the panel's centroid always lands at viewport center, regardless of
scale. Push-ins then zoom around the panel center instead of drifting.

Base scale per panel: `min(viewport_w / panel_w, viewport_h / panel_h)`.

### 3.5 Timeline pattern (per panel)

```
transition:  tl.to("#page", focus(PANELS.pN), {duration: 0.6-1.0})
reveal:      tl.to("#mask<N>", {opacity: 0, duration: 0.35-0.50})
push-in:     tl.to("#page", focus(PANELS.pN, 1.05-1.07), {duration: 1-3})
```

- **P1 (cold open):** skip transition; page starts parked at P1, veil fades,
  mask explodes away (0.15 s), stage shakes, slight scale kick.
- **Middle panels:** linger on important ones (P2), snap through
  less-important ones (P3, P4).
- **Final splash (P5):** bigger is better.
  - Flash overlay (`opacity 0→0.85→0` over 0.4 s)
  - Triple scale punch: zoom kick + bounce back
  - Heavy screen shake (10 keyframes, magnitude 28-30 px)
  - Cap final push-in scale at 1.00–1.03 so side text stays in frame

### 3.6 Full-screen overlays

- `#veil`: opaque black at t=0, fades during cold open. Covers whatever's
  behind the masks during the pre-reveal phase.
- `#flash`: white rect, opacity 0 normally. Spiked during DOMM-style
  impacts.
- `#vignette`: subtle radial darkening, always on. Adds atmosphere.

### 3.7 Render

```bash
cd manga-motion && npx hyperframes render
```

~15–20 s on laptop, produces 1080×1920 30fps MP4, ~15 MB for 18 s.

### 3.8 Render reliability — the killswitch

Hyperframes occasionally stalls during frame capture (silent hang, no output,
no process death). Wrap render invocations with a hard timeout:

```bash
pkill -9 -f "hyperframes render" 2>/dev/null; sleep 1
cd manga-motion && \
  (npx hyperframes render 2>&1 | tail -3) & PID=$!
(sleep 60 && kill -9 $PID 2>/dev/null && echo "--- TIMED OUT ---") & WATCHDOG=$!
wait $PID 2>/dev/null; RC=$?
kill $WATCHDOG 2>/dev/null; wait $WATCHDOG 2>/dev/null
exit $RC
```

Normal render is 15–25 s for 18 s of content. If a render doesn't complete in
60 s, it's stalled — kill and retry. Retries almost always succeed.

---

## 4. Stage 4: Audio and pacing (ElevenLabs + timeline craft)

Audio carries as much weight as the visuals. A silent motion comic with
pixel-perfect masks is boring; a 70%-accurate masked comic with good VO and
SFX is electric. Budget accordingly.

### 4.1 Voice casting

**Workspace voices vs the shared library.**  The `/v1/voices` endpoint only
returns the 20-ish voices in your workspace. Use `/v1/shared-voices` to search
the full community library (thousands of voices). Filter by `gender`, `age`,
`use_cases=characters_animation`, or search by keyword (`anime`, `intense`,
`hero`).

**Cast per character, not per project.**  Keep a voice-ID-to-character map.
For Pizza Blitz: Blitz (`f3ipu…`) = protagonist, Timid Tim (`Dqir5…`) =
confused customer. Reuse across chapters so characters stay sonically
consistent the same way reference-image character sheets keep them visually
consistent.

**Preview first, cast second.**  Each shared voice has a `preview_url` in its
voice record. Download and listen before generating a full chapter's worth of
lines — casting the wrong voice means regenerating everything.

### 4.2 Voice delivery per line

Four knobs on `voice_settings`:

| Knob | Low | High |
|------|-----|------|
| `stability` | more emotional variation (dramatic) | consistent, controlled |
| `similarity_boost` | drifts from voice | stays in character (usually 0.8) |
| `style` | neutral | exaggerated, performative |
| `use_speaker_boost` | — | louder, closer-mic feel |

**Tune per line, not per character.** Same character speaking different lines
needs different settings. A shout line needs `stability` ≈ 0.15–0.30 and
`style` ≈ 0.80–1.00. A whisper needs `stability` ≈ 0.60–0.70 and
`style` ≈ 0.15–0.30.

**Reference settings that worked:**

```python
# Reverent, low-intensity monologue
("promise",    0.35, 0.80, 0.55, True)
# Intense steely resolve (emphasize "ZERO")
("mistakes",   0.30, 0.80, 0.80, True)
# Smug, controlled
("calculated", 0.55, 0.80, 0.55, True)
# MAX-volume shout
("arrived",    0.25, 0.85, 0.90, True)
```

### 4.3 v3 audio tags are a cheat code

With `model_id: "eleven_v3"`, you can prefix text with directive tags:

```
[whispers] Exactly as I calculated.
[shouts] Your pizza... [yells] HAS ARRIVED!!!
[intense] ZERO mistakes.
[confused] Huh?
[smug] Exactly as I calculated.
```

Available tags include: `[whispers]`, `[shouts]`, `[yells]`, `[intense]`,
`[angry]`, `[sad]`, `[happy]`, `[sarcastic]`, `[confused]`, `[smug]`,
`[laughs]`, `[sighs]`, `[gasps]`.

Tags compound with `voice_settings` — use tags for the *what* (whispered vs
shouted), settings for the *how much* (how dramatic/controlled).

### 4.4 Text itself matters

Even without audio tags, punctuation and casing steer delivery:

- **CAPS** signal emphasis: `ZERO mistakes` punches harder than
  `zero mistakes`.
- **Mid-sentence ellipses cause pauses**, which sometimes sound weird
  (`Exactly... as I calculated` produced an awkward beat in the middle).
  Prefer end-of-clause ellipses, or remove them and use audio tags for
  restraint.
- **Caps + exclamation points** on a single word (`HAS ARRIVED!!!`) triggers
  yelling without needing `[yells]`.

### 4.5 Single take > splicing

Two sentences in one TTS call (e.g., `"This isn't just food... It's a
promise. One box. One customer. ZERO mistakes."`) gives natural
inter-sentence breath. Two separate clips concatenated sound *robotic* because
there's no shared prosody across the boundary. Always prefer one TTS call per
"spoken block," even if the block has multiple sentences.

### 4.6 Speed adjustment via ffmpeg

ElevenLabs has a `speed` parameter in `voice_settings` but it doesn't seem to
apply on all models (v3 in particular). For reliable tempo control, do it in
post with `ffmpeg atempo`:

```bash
ffmpeg -y -i in.mp3 -filter:a "atempo=1.12" out.mp3
```

Preserves pitch. `atempo=1.10–1.20` is the useful range for "make this line
snappier without sounding chipmunked." Beyond 1.25 starts to sound unnatural.

### 4.7 Sound effects

**Prompt-driven SFX via `/v1/sound-generation`.**  Text prompt + duration
(max ~22 s). Works well for short ambient stings and impacts.

```python
POST /v1/sound-generation
{
  "text": "heavy bass kick, cinematic drop, manga panel-break impact",
  "duration_seconds": 1.5,
  "prompt_influence": 0.5
}
```

**User-provided SFX > AI-generated for iconic sounds.**  We generated
`sfx_p5.mp3` ("heavy impact") via API and it was fine but not punchy. A
custom `boom.mp3` the user dropped in the folder was dramatically better.
Invest in a small library of signature SFX (impact, footsteps, whoosh) and
reuse. AI-generated is fine for ambient/atmospheric stuff where quality
doesn't carry the emotional beat.

**Layer for density.**  P3's scooter-action panel uses both a synthesized
whoosh AND a real running-footsteps clip layered together. Single SFX feels
flat; two layered SFX feels kinetic.

**Volume hierarchy.**  Rough defaults:
- Voiceover (main line): `1.0`
- Ambient / establishing SFX (wind, drone): `0.7–0.8`
- Percussive hits (boom, crash): `0.9–1.0`
- Secondary layer (footsteps under a whoosh): `0.9`

### 4.8 Timing and pacing rules

The craft that makes it feel tight.

**The "0.3s early" rule.**  Voiceover for the NEXT panel starts 0.3 s BEFORE
the camera finishes panning to it. The ear hears the line beginning over the
tail of the camera move, which pulls the eye forward into the new panel.
Feels cinematic. Zero overlap feels staccato.

**Pan duration: 0.6 s.**  Sweet spot for between-panel transitions. 0.9 s
drags. 0.4 s feels jerky.

**Pan next when sounds finish.**  If a panel has no voice, pan away when the
SFX ends. If it has voice, pan away ~0–0.3 s after the last word. Silence
between panels is where tension leaks.

**Panel hold = fits the audio.**  Don't invent hold times; derive them.
A panel with a 3 s line + 0.3 s breath = 3.3 s hold. A panel with a 2 s SFX
= 2 s hold.

**Cold open without a title card.**  Skip "CHAPTER 1 / THE OATH" entirely.
Start on panel 1 with an impact: mask slam + scale kick + screen shake + SFX,
all at t=0. The title-card pattern feels like TV, the cold-open pattern feels
like shonen.

**Comedic timing: <1 s between punchline setups.**  The setup-punch joke (P4
smug "Exactly as I calculated" → P5 shouted "YOUR PIZZA HAS ARRIVED!!!") only
lands if the gap is short. Aim for 0.3–0.5 s dead air between them.

**Sync the slam with the SHOUT, not the camera.**  For climactic panels,
fire flash + mask-fade + scale-punch + shake + impact SFX all at the exact
instant the voiceover starts its peak moment. P5's reveal lands on the first
syllable of "YOUR PIZZA" — all six events simultaneous. Anything less
synchronized feels wrong.

**Final hold outlasts the SFX.**  If boom SFX runs for 4 s, hold on P5 for
at least 4 s. Let it decay. Cutting on the SFX tail feels abrupt.

### 4.9 Hyperframes audio attributes

```html
<audio id="vo-promise" class="clip"
       data-start="2.20" data-duration="2.5"
       data-track-index="5" data-volume="1.0"
       src="audio/promise.mp3"></audio>
```

- `class="clip"` — required (runtime visibility lifecycle)
- `id` — required and unique (lint error otherwise)
- `data-start` — seconds from composition start
- `data-duration` — slightly longer than actual audio for safety (e.g.,
  2.5s for a 2.44s clip)
- `data-track-index` — use 5+ for audio by convention (3 = veil, 4 = flash,
  2 = title, 1 = main stage)
- `data-volume` — 0 to 1

### 4.10 Iteration workflow

Voice generation is nondeterministic; TTS re-rolls return different takes.
Practical workflow:

1. **Batch-generate initial takes** from a single script with per-line
   settings.
2. **Listen, flag the duds.**  Usually 1–2 lines out of 5 need redelivery.
3. **Regenerate only the flagged lines.**  Preserve approved takes — don't
   re-run the whole batch (you'll lose good ones to randomness).
4. **Iterate on prompt + settings + text** until each line lands.
5. **Post-process with ffmpeg** (speed, trim, fade) rather than re-generating
   when the delivery is right but the pacing is off.

---

## 5. Things we tried that didn't work (and why)

Keeping a record so we don't re-litigate.

- **Classical CV-only panel detection** (threshold + flood-fill). Works
  on pages with clean bright gutters. Fails on pages with dark/tonal
  gutters or character figures breaking the gutter. No parameter tune
  fixes all cases.
- **LLM-only panel detection** (ask for polygons directly). Topology
  always right; coordinates 50–200 px off, always as rectangles (ignores
  angled gutters).
- **Hough transform to find gutters directly.** Catches tons of internal
  line art (speed lines, architecture) as "lines." Even with aggressive
  filtering, lands far off.
- **Image-gen mask with no frame constraint (v1).** Model drew a black
  frame around the page, shrinking bleed panels.
- **Convex hull of panel contours.** Destroys concavity — any panel with
  a character figure breaking the frame collapses.
- **Wide-scan snap with "outermost signal above threshold" selection.**
  Every column has SOME signal in a busy manga page; selection just
  expands to the scan window's edge.
- **Wide-scan snap with "strongest peak" selection.** Picks strong in-panel
  features (character contours) over weak real gutters.
- **Average-two-panels' snapped lines.** Only works if both panels'
  snaps landed near the true gutter. If both were pulled by the same
  misplaced mask, the average is just two-wrongs-make-a-wrong.

The combination that works: **narrow-scan per-sample snap + shared-gutter
wide-scan averaged snap.** Narrow catches pixel-precise edges when the
mask is close; wide-averaged catches real gutters when the mask is far
off, using gutter-length consistency to reject noise.

**Audio/pacing failure modes:**

- **Concatenating two TTS clips for two sentences.**  Sounds robotic — no
  shared prosody across the boundary. Single-take VO with both sentences
  sounds human.
- **The `speed` parameter in ElevenLabs `voice_settings`.**  Didn't actually
  affect output duration on `eleven_v3` in our tests. Use ffmpeg `atempo`
  post-processing instead.
- **"Outermost signal above threshold" scan in the snap** (reused failure
  mode from CV land, mentioned here for completeness). Same pathology:
  a busy page has signal everywhere, so "outermost above threshold"
  expands to the scan-window edge.
- **AI-generated SFX for climactic impacts.**  Fine for ambient/atmospheric,
  too generic for iconic moments. Use hand-picked SFX for the signature
  sounds (boom, footsteps, whooshes) and AI for filler.
- **Hyperframes render without a killswitch.**  Silent stalls happen.
  Wrap in a timeout + kill + retry loop.
- **Using opaque title cards.**  Felt like a TV intro, not a cinematic cold
  open. Removed entirely — motion comic starts on the first panel.

---

## 6. File map

```
manga-motion/
├── PIPELINE.md                  # this doc
├── gen_panel_mask_v2.py         # stage 2a: mask generation (GPT Image 2)
├── snap_mask.py                 # stage 2b: mask refinement (snap to gutters)
├── overlay_mask.py              # diagnostic: overlay mask on page
├── gen_voiceover.py             # stage 3a: initial VO batch
├── gen_voiceover_v2.py          # stage 3b: regens + SFX via sound-generation
├── index.html                   # stage 4: Hyperframes composition
├── page{N}.png                  # page renders
├── page{N}_mask_v2.png          # binary panel-fills masks
├── page{N}_v2_snapped_panels.json  # snapped polygons (per panel)
├── page{N}_v2_snapped_overlay.png  # verification overlay
├── audio/
│   ├── p2_lines.mp3             # combined single-take VO for P2
│   ├── calculated.mp3           # per-panel voice line
│   ├── arrived.mp3              # (etc.)
│   ├── huh.mp3                  # customer reaction (Timid Tim)
│   ├── sfx_p1.mp3               # AI-generated SFX
│   ├── sfx_p3.mp3               # AI-generated SFX
│   ├── runningfootsteps.mp3     # user-supplied SFX
│   └── boom.mp3                 # user-supplied SFX (the good one)
└── renders/                     # MP4 outputs
```
