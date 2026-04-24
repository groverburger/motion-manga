"""Generate voiceover lines for page 1's pizza-guy protagonist via ElevenLabs.

Uses the "Blitz" custom voice — tuned specifically for the Pizza Blitz
protagonist.

Output: audio/*.mp3 files, one per line. Each line uses per-line voice
settings tuned to its delivery intensity.
"""
import pathlib, requests, sys

KEY = pathlib.Path.home().joinpath("Documents/elevenlabs_api_key.txt").read_text().strip()
VOICE_ID = "f3ipuCDocGuYEK3f9JwC"    # Blitz (custom)
MODEL_ID = "eleven_multilingual_v2"
OUT_DIR = pathlib.Path(__file__).parent / "audio"
OUT_DIR.mkdir(exist_ok=True)

# (slug, text, stability, similarity_boost, style, speaker_boost)
# - Lower stability = more emotional variation (more dramatic)
# - Higher style    = more exaggerated delivery
# - speaker_boost   = louder/closer-mic feel
LINES = [
    # Panel 2: inner monologue over the pizza box — quiet, reverent intensity
    ("promise",
     "This isn't just food... It's a promise.",
     0.35, 0.80, 0.55, True),
    # Panel 2: steely clipped resolve
    ("mistakes",
     "One box. One customer. Zero mistakes.",
     0.50, 0.80, 0.40, True),
    # Panel 4: cold satisfied thought
    ("calculated",
     "Exactly as I calculated.",
     0.45, 0.80, 0.55, True),
    # Panel 5: explosive, shouted climax
    ("arrived",
     "YOUR PIZZA... HAS ARRIVED!!!",
     0.25, 0.85, 0.90, True),
]

def synth(text, stability, similarity_boost, style, speaker_boost):
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
        headers={
            "xi-api-key": KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={
            "text": text,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": speaker_boost,
            },
        },
        timeout=120,
    )
    if r.status_code != 200:
        print(f"  ERROR {r.status_code}: {r.text[:400]}")
        return None
    return r.content

for slug, text, stab, sim, style, boost in LINES:
    out = OUT_DIR / f"{slug}.mp3"
    print(f"generating {slug}: {text!r}")
    audio = synth(text, stab, sim, style, boost)
    if audio:
        out.write_bytes(audio)
        print(f"  saved {out} ({len(audio)/1024:.1f} KB)")
