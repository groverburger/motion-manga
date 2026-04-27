"""Re-generate the two lines that needed revision:
   - calculated (P4): more whispery
   - arrived    (P5): more exclamation
Plus generate three SFX for panels 1, 3, 5 via the sound-generation API.

Preserves existing audio/promise.mp3 and audio/mistakes.mp3 untouched.
"""
import pathlib, requests, sys

KEY = pathlib.Path.home().joinpath("Documents/elevenlabs_api_key.txt").read_text().strip()
VOICE_ID = "f3ipuCDocGuYEK3f9JwC"    # Blitz
OUT = pathlib.Path(__file__).parent / "audio"
OUT.mkdir(exist_ok=True)

def tts(text, model, stability, sim, style, speaker_boost):
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
        headers={"xi-api-key": KEY, "Content-Type": "application/json",
                 "Accept": "audio/mpeg"},
        json={
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": sim,
                "style": style,
                "use_speaker_boost": speaker_boost,
            },
        },
        timeout=120,
    )
    return r

def sfx(text, duration):
    r = requests.post(
        "https://api.elevenlabs.io/v1/sound-generation",
        headers={"xi-api-key": KEY, "Content-Type": "application/json",
                 "Accept": "audio/mpeg"},
        json={
            "text": text,
            "duration_seconds": duration,
            "prompt_influence": 0.5,
        },
        timeout=180,
    )
    return r

# ---- Voice regens ----
# Try v3 first (supports [whispers], [shouts] audio tags). Fall back to v2 with
# extreme settings if v3 isn't available on this account.
def save_voice(slug, text_v3, text_v2_fallback, v2_params):
    print(f"[voice] {slug}")
    r = tts(text_v3, "eleven_v3", 0.4, 0.80, 0.55, True)
    if r.status_code != 200:
        print(f"  v3 failed ({r.status_code}), using v2 with extreme settings")
        r = tts(text_v2_fallback, "eleven_multilingual_v2", *v2_params)
    if r.status_code != 200:
        print(f"  v2 also failed: {r.status_code} {r.text[:300]}")
        return
    out = OUT / f"{slug}.mp3"
    out.write_bytes(r.content)
    print(f"  saved {out} ({len(r.content)/1024:.1f} KB)")

# P4: whisper. v3 audio tag; v2 fallback uses lower stability + slower energy
save_voice(
    "calculated",
    text_v3="[whispers] Exactly... as I calculated.",
    text_v2_fallback="Exactly... as I calculated.",
    v2_params=(0.70, 0.80, 0.15, False),   # high stability + low style = quiet
)

# P5: exclamation, maximum exuberance
save_voice(
    "arrived",
    text_v3="[shouts] Your pizza... [yells] HAS ARRIVED!!!",
    text_v2_fallback="Your pizza... HAS ARRIVED!!!",
    v2_params=(0.15, 0.85, 1.0, True),    # very low stability + max style
)

# ---- SFX ----
SFX_CLIPS = [
    # slug, text prompt, duration seconds
    ("sfx_p1",
     "Dramatic cinematic sting — deep ominous bass drone with a single rising tense pulse, "
     "anime-style intensity build. Short, impactful, no music.",
     2.5),
    ("sfx_p3",
     "High-energy motion whoosh — scooter engine zooming past, wind rushing, "
     "anime-style ZSSSHH motion sound, urgency and speed.",
     2.0),
    ("sfx_p5",
     "Massive impact — heavy bass kick, deep booming thud with low-end rumble, "
     "a single concussive percussive hit like a cinematic drop or manga panel-break, "
     "no music.",
     1.5),
]

for slug, text, dur in SFX_CLIPS:
    print(f"[sfx] {slug} ({dur}s)")
    r = sfx(text, dur)
    if r.status_code != 200:
        print(f"  FAILED {r.status_code}: {r.text[:300]}")
        continue
    out = OUT / f"{slug}.mp3"
    out.write_bytes(r.content)
    print(f"  saved {out} ({len(r.content)/1024:.1f} KB)")
