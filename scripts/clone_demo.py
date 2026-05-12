"""Zero-shot voice clone using XTTS v2.

Picks a reference WAV (configurable) and synthesizes several sample sentences
into output/clones/. First run downloads XTTS v2 (~2 GB).
"""
import os
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from TTS.api import TTS

ROOT = Path(__file__).resolve().parent.parent
REF_WAV = ROOT / "data" / "wavs" / "voice_015.wav"  # 14.9s mid-length clip
OUT_DIR = ROOT / "output" / "clones"
MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

SAMPLES = [
    ("hello.wav", "Hello, this is my cloned voice. How does it sound?"),
    ("pangram.wav", "The quick brown fox jumps over the lazy dog."),
    ("longer.wav", "I have been working on a voice cloning project, and I think it is starting to sound like me."),
]


def main() -> None:
    if not REF_WAV.exists():
        raise SystemExit(f"Reference WAV not found: {REF_WAV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID}…")
    tts = TTS(MODEL_ID, progress_bar=True)

    print(f"Reference: {REF_WAV.name}")
    for fname, text in SAMPLES:
        out_path = OUT_DIR / fname
        print(f"\nGenerating: {fname}")
        print(f"  Text: {text}")
        tts.tts_to_file(
            text=text,
            speaker_wav=str(REF_WAV),
            language="en",
            file_path=str(out_path),
        )
        print(f"  → {out_path}")

    print(f"\nDone. {len(SAMPLES)} clips in {OUT_DIR}")


if __name__ == "__main__":
    main()
