# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This repo is currently a **scoping document only** (`README.md`). No source code, dependencies, build system, or training scripts exist yet. The first concrete task will be setting up the Python environment and installing Coqui TTS.

When the user asks to "start" or "build", treat that as bootstrapping the project from scratch — do not assume any tooling, virtualenv, or dataset layout is already in place.

## Goal

Build a text-to-speech voice from scratch using open-source tools, on a single-GPU/local-workstation scale (~1 hour of audio, hours of training).

## Intended Stack

The README commits to a specific toolchain. Prefer these over alternatives unless the user explicitly redirects:

- **Coqui TTS** as the training/inference framework
- **VITS** as the default model (end-to-end: acoustic model + vocoder in one)
- **XTTS v2** if voice cloning is needed
- **Whisper** for auto-transcribing collected audio into a training corpus
- **ffmpeg** + **librosa / torchaudio** for audio preprocessing

Target: Python 3.9+, NVIDIA GPU with CUDA strongly preferred.

## Pipeline Architecture

The full pipeline has three logical stages, even though VITS collapses (2) and (3):

1. **Text processor** — text → phonemes / linguistic features
2. **Acoustic model** — features → mel-spectrograms
3. **Vocoder** — spectrograms → waveform

End-to-end workflow the repo is being built around:

1. Record / gather clean audio
2. Transcribe with Whisper
3. Clean & normalize (consistent volume, trim silence, denoise)
4. Format as a dataset (audio file ↔ transcript pairs)
5. Configure a VITS recipe via Coqui TTS
6. Train
7. Inference

When adding code, organize it along these stages rather than as one monolithic script.
