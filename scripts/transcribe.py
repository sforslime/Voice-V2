"""Transcribe all WAVs in data/wavs/ with Whisper, write Coqui LJSpeech-format metadata.csv."""
import csv
import sys
from pathlib import Path

import whisper

ROOT = Path(__file__).resolve().parent.parent
WAV_DIR = ROOT / "data" / "wavs"
META_PATH = ROOT / "data" / "metadata.csv"
MODEL_SIZE = "medium"  # tiny | base | small | medium | large-v3
LANGUAGE = "en"  # force language to avoid mis-detection on short/noisy clips


def main() -> None:
    wavs = sorted(WAV_DIR.glob("*.wav"))
    if not wavs:
        sys.exit(f"No WAVs in {WAV_DIR}")

    print(f"Loading Whisper '{MODEL_SIZE}'…")
    model = whisper.load_model(MODEL_SIZE)

    rows = []
    for i, wav in enumerate(wavs, 1):
        print(f"[{i}/{len(wavs)}] {wav.name}")
        result = model.transcribe(str(wav), fp16=False, language=LANGUAGE)
        text = result["text"].strip()
        rows.append((wav.stem, text, text))
        print(f"    → {text[:80]}{'…' if len(text) > 80 else ''}")

    with META_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {META_PATH}")


if __name__ == "__main__":
    main()
