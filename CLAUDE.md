# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Phase 0 (environment setup) is complete. Coqui TTS + Whisper installed in a local venv, MPS-enabled torch verified, pre-trained VITS sanity-check passed (see `output/sanity_check.wav`). No custom training code yet.

## Target Hardware

Local Mac, Apple Silicon (M4, 24GB unified). All training and inference is intended to run on-device via PyTorch MPS. No cloud GPU planned.

Implications:
- Coqui's MPS support has gaps — expect some ops to fall back to CPU. Don't assume CUDA-style throughput.
- Prefer **XTTS v2 fine-tuning** (hours) over **VITS-from-scratch** (days) as the first training run. VITS-from-scratch is a stretch goal once the pipeline is proven end-to-end.

## Environment

- Python: 3.11 (via Homebrew `python@3.11`). Coqui TTS does **not** support Python 3.13.
- Virtualenv: `.venv/` at repo root. Always invoke as `.venv/bin/python` / `.venv/bin/pip`.
- Requirements pinned in `requirements.txt` (generated via `pip freeze`).
- System deps installed via Homebrew: `ffmpeg`, `espeak-ng` (required for the VITS phonemizer; install fails at inference time without it).

### Dependency gotchas (learned the hard way)

- `coqui-tts` does **not** declare `torch` as a hard dep — install `torch torchaudio` explicitly.
- `coqui-tts 0.27.x` requires `transformers < 5`. Pip will pull transformers 5.x by default, which removes `isin_mps_friendly` and breaks the XTTS import path. Pin `transformers<5`.
- Torch ≥ 2.9 requires `torchcodec` for audio IO. Install via `pip install "coqui-tts[codec]"`.

## Goal

Build a text-to-speech voice from scratch using open-source tools. Pipeline:

1. Record / gather clean audio
2. Transcribe with Whisper
3. Clean & normalize (volume, silence trim, denoise)
4. Format as a Coqui-style dataset (`wavs/` + `metadata.csv`)
5. Fine-tune XTTS v2 (Phase 1) or train VITS from scratch (later)
6. Inference

A TTS system has three logical stages — text → features (phonemizer), features → spectrogram (acoustic model), spectrogram → waveform (vocoder). VITS and XTTS both collapse the last two into one end-to-end model.

## Layout (current)

```
.venv/             # Python 3.11 virtualenv (gitignored)
output/            # Generated audio + checkpoints (gitignored)
requirements.txt   # Frozen pip deps
CLAUDE.md
README.md
.gitignore
```

`data/`, `dataset/`, `wavs/`, `checkpoints/`, `runs/` are reserved names and pre-ignored in `.gitignore`. Audio files (`*.wav`, `*.mp3`, etc.) are ignored except under `samples/`.

## Common commands

```bash
# Activate / use the venv
.venv/bin/python …
.venv/bin/pip …

# Run a pre-trained voice (sanity check)
.venv/bin/python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/vits').tts_to_file(text='hello', file_path='output/test.wav')"

# Play a generated wav
afplay output/sanity_check.wav

# Re-create env from scratch
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
