"""Run the fine-tuned XTTS v2 model locally.

Expects models/finetuned/ to contain:
    best_model.pth     (fine-tuned GPT checkpoint, from Colab)
    config.json        (training config)
    vocab.json         (tokenizer)

Generates samples in output/finetuned/.
"""
import os
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import soundfile as sf
import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "finetuned"
OUT_DIR = ROOT / "output" / "finetuned"
REF_WAV = ROOT / "data" / "wavs" / "voice_015.wav"

SAMPLES = [
    ("hello.wav", "Hello, this is my fine-tuned voice. Does this sound more like me now?"),
    ("pangram.wav", "The quick brown fox jumps over the lazy dog."),
    ("longer.wav", "I have been working on a voice cloning project, and I think it is starting to sound like me."),
]


def main() -> None:
    for name in ("best_model.pth", "config.json", "vocab.json"):
        path = MODEL_DIR / name
        if not path.exists():
            raise SystemExit(f"Missing {path}. Download from Drive into {MODEL_DIR}/")

    if not REF_WAV.exists():
        raise SystemExit(f"Reference WAV missing: {REF_WAV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    config = XttsConfig()
    config.load_json(str(MODEL_DIR / "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(MODEL_DIR / "best_model.pth"),
        vocab_path=str(MODEL_DIR / "vocab.json"),
        use_deepspeed=False,
    )
    model.to(device)

    for fname, text in SAMPLES:
        out_path = OUT_DIR / fname
        print(f"\n{fname}: {text}")
        out = model.synthesize(
            text,
            config,
            speaker_wav=str(REF_WAV),
            gpt_cond_len=3,
            language="en",
        )
        sf.write(str(out_path), out["wav"], 24000)
        print(f"  → {out_path}")

    print(f"\nDone. {len(SAMPLES)} clips in {OUT_DIR}")


if __name__ == "__main__":
    main()
