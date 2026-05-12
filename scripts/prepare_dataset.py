"""Build XTTS-ready dataset from data/wavs/.

For each WAV in data/wavs/, run Whisper to get sentence segments with timestamps,
slice the audio at those boundaries into 2-15s clips, and write a Coqui LJSpeech
format dataset to data/dataset/ (wavs/ + metadata.csv).
"""
import csv
import sys
from pathlib import Path

import soundfile as sf
import whisper

ROOT = Path(__file__).resolve().parent.parent
SRC_WAV_DIR = ROOT / "data" / "wavs"
OUT_DIR = ROOT / "data" / "dataset"
OUT_WAV_DIR = OUT_DIR / "wavs"
META_PATH = OUT_DIR / "metadata.csv"

MODEL_SIZE = "medium"
LANGUAGE = "en"
MIN_DUR = 3.5   # seconds — XTTS needs >=3s for conditioning slice; pad for safety
MAX_DUR = 11.0  # seconds — XTTS max_wav_length is ~11.6s; pad for safety


def main() -> None:
    wavs = sorted(SRC_WAV_DIR.glob("*.wav"))
    if not wavs:
        sys.exit(f"No WAVs in {SRC_WAV_DIR}")

    OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Whisper '{MODEL_SIZE}'…")
    model = whisper.load_model(MODEL_SIZE)

    rows = []
    clip_idx = 0
    kept_dur = 0.0
    dropped_short = 0
    dropped_long = 0

    for i, wav_path in enumerate(wavs, 1):
        audio, sr = sf.read(str(wav_path))
        result = model.transcribe(str(wav_path), fp16=False, language=LANGUAGE)
        segs = result["segments"]
        print(f"[{i}/{len(wavs)}] {wav_path.name} → {len(segs)} segments")

        for seg in segs:
            start, end = float(seg["start"]), float(seg["end"])
            text = seg["text"].strip()
            dur = end - start

            if dur < MIN_DUR:
                dropped_short += 1
                continue
            if dur > MAX_DUR:
                dropped_long += 1
                continue
            if not text:
                continue

            clip_idx += 1
            clip_name = f"clip_{clip_idx:04d}"
            out_path = OUT_WAV_DIR / f"{clip_name}.wav"

            slice_audio = audio[int(start * sr):int(end * sr)]
            sf.write(str(out_path), slice_audio, sr, subtype="PCM_16")

            rows.append((clip_name, text, text))
            kept_dur += dur

    with META_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerows(rows)

    print(f"\nKept {len(rows)} clips ({kept_dur/60:.2f} min)")
    print(f"Dropped: {dropped_short} short (<{MIN_DUR}s), {dropped_long} long (>{MAX_DUR}s)")
    print(f"Dataset → {OUT_DIR}")


if __name__ == "__main__":
    main()
