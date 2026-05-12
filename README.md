# Digital Voice Project

## Goal
Build a text-to-speech voice from scratch using open-source tools.

---

## How It Works (Overview)

A TTS system has three stages:
1. **Text Processor** – converts text into phonemes / linguistic features
2. **Acoustic Model** – converts features into audio spectrograms
3. **Vocoder** – converts spectrograms into actual audio waveforms

Modern end-to-end models like **VITS** combine steps 2 and 3 into one.

---

## Recommended Stack

| Tool | Purpose |
|------|---------|
| **Coqui TTS** | Open-source TTS framework (wraps VITS, XTTS, etc.) |
| **VITS** | End-to-end acoustic model + vocoder |
| **XTTS v2** | Voice cloning built-in (also by Coqui) |
| **Whisper** | Auto-transcribe audio recordings |
| **ffmpeg** | Audio processing & conversion |
| **librosa / torchaudio** | Audio preprocessing in Python |

---

## Requirements

- Python 3.9+
- GPU strongly recommended (NVIDIA with CUDA)
- 1–20+ hours of clean audio recordings (more = better quality)

---

## Steps to Build

1. **Record / gather audio data** – clean, quiet, consistent recordings
2. **Transcribe** – use Whisper to auto-generate transcripts
3. **Clean & normalize audio** – consistent volume, trim silence, remove noise
4. **Format dataset** – pair each audio file with its transcript
5. **Configure model** – start with VITS via Coqui TTS
6. **Train** – hours to days depending on GPU
7. **Run inference** – generate speech from new text

---

## Key Resources

- [Coqui TTS GitHub](https://github.com/coqui-ai/TTS)
- [VITS Paper](https://arxiv.org/abs/2106.06103)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## Notes
- A personal voice model (~1 hr of data, single GPU) can train in a few hours
- Production-quality voices need much more data and compute
- Next step: set up environment and install Coqui TTS
