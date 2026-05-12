"""Fine-tune XTTS v2 GPT on a Coqui LJSpeech-style dataset.

Parameterized via environment variables so the same script can run locally or in
Colab.

Required env vars:
    DATASET_PATH   Folder containing wavs/ and metadata.csv
    OUTPUT_PATH    Where to write checkpoints / logs (use a Drive path in Colab)

Optional env vars:
    BATCH_SIZE     Default 3 (good for 16-24GB GPUs)
    GRAD_ACCUM     Default 16 (effective batch = BATCH_SIZE * GRAD_ACCUM)
    EPOCHS         Default 20
    LANGUAGE       Default "en"
    SAVE_STEP      Default 500 (checkpoints survive Colab disconnects)
    LR             Default 5e-6
    REF_WAV        Speaker reference WAV for periodic test samples. Defaults to
                   the first WAV in DATASET_PATH/wavs/.
"""
import os
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager


# ---------- config from env ----------
DATASET_PATH = Path(os.environ["DATASET_PATH"]).resolve()
OUTPUT_PATH = Path(os.environ["OUTPUT_PATH"]).resolve()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "3"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
LANGUAGE = os.environ.get("LANGUAGE", "en")
SAVE_STEP = int(os.environ.get("SAVE_STEP", "500"))
LR = float(os.environ.get("LR", "5e-6"))

META_FILE = DATASET_PATH / "metadata.csv"
WAV_DIR = DATASET_PATH / "wavs"
assert META_FILE.exists(), f"Missing {META_FILE}"
assert WAV_DIR.exists(), f"Missing {WAV_DIR}"

REF_WAV = os.environ.get("REF_WAV") or str(sorted(WAV_DIR.glob("*.wav"))[0])


# ---------- download XTTS v2 base + DVAE ----------
CHECKPOINTS_DIR = OUTPUT_PATH / "xtts_v2_base"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "dvae.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
    "mel_stats.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth",
    "vocab.json": "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
    "model.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
}
missing = [url for name, url in FILES.items() if not (CHECKPOINTS_DIR / name).exists()]
if missing:
    print(" > Downloading XTTS v2 base files…")
    ModelManager._download_model_files(missing, str(CHECKPOINTS_DIR), progress_bar=True)

DVAE_CHECKPOINT = str(CHECKPOINTS_DIR / "dvae.pth")
MEL_NORM_FILE = str(CHECKPOINTS_DIR / "mel_stats.pth")
TOKENIZER_FILE = str(CHECKPOINTS_DIR / "vocab.json")
XTTS_CHECKPOINT = str(CHECKPOINTS_DIR / "model.pth")


def main() -> None:
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="voice_v2",
        path=str(DATASET_PATH),
        meta_file_train=str(META_FILE),
        language=LANGUAGE,
    )

    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6s
        min_conditioning_length=66150,   # 3s
        debug_loading_failures=False,
        max_wav_length=255995,           # ~11.6s
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    config = GPTTrainerConfig(
        output_path=str(OUTPUT_PATH),
        model_args=model_args,
        run_name="voice_v2_xtts_ft",
        project_name="voice_v2",
        run_description="XTTS v2 fine-tune on voice_v2 dataset",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=2,
        eval_split_max_size=64,
        epochs=EPOCHS,
        print_step=25,
        plot_step=100,
        log_model_step=SAVE_STEP,
        save_step=SAVE_STEP,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=LR,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {"text": "Hello, this is my cloned voice. How does it sound now?",
             "speaker_wav": [REF_WAV], "language": LANGUAGE},
            {"text": "The quick brown fox jumps over the lazy dog.",
             "speaker_wav": [REF_WAV], "language": LANGUAGE},
        ],
    )

    model = GPTTrainer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=0.05,
    )

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=GRAD_ACCUM,
        ),
        config,
        output_path=str(OUTPUT_PATH),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
