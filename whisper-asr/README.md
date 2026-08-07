# 🎤 Whisper ASR — Speech-to-Text Fine-Tune

Fine-tunes **`openai/whisper-tiny`** (39M params) on a small English ASR corpus —
the "hear the user" stage of the voice-assistant pipeline. Because a voice
assistant needs to transcribe user speech before anything else can happen, this
project fine-tunes Whisper on clean, short utterances so it works well with
microphone input.

| | |
|---|---|
| Base model | `openai/whisper-tiny` (39M) |
| Task | Speech → Text |
| Technique | Full Seq2Seq fine-tune (encoder + decoder) |
| Dataset | LibriSpeech clean subset (see below) |
| VRAM | ~4 GB |

---

## Dataset

This project fine-tunes on **LibriSpeech**, loaded on the fly from Hugging Face
via the `datasets` library — no manual download needed:

- `train-clean-100` (a slice, ~10 hours) for training
- `test-clean` for evaluation

If you have your own audio (e.g. voice-assistant recordings), swap
`load_dataset` in `train_whisper.py` for your own loader — the rest of the
pipeline is unchanged.

## Training

```bash
python train_whisper.py
```

The script:

1. Loads Whisper's **feature extractor**, **tokenizer**, and **processor** from
   the base model.
2. Prepares examples by resampling audio to 16 kHz and computing log-mel
   features (`WhisperFeatureExtractor`).
3. Tokenizes transcripts and **masks the decoder input ids** (`labels = -100`)
   so only the transcript is supervised.
4. Trains with the `Seq2SeqTrainer` from `transformers`, evaluating **WER** on
   the test split after training.

### Expected results

`whisper-tiny` is the smallest Whisper — the goal here is a **working
end-to-end recipe**, not SOTA. Expect WER around **15–25%** on `test-clean`
after fine-tuning; upgrading to `whisper-small`/`whisper-base` is a one-line
change (`MODEL_NAME`) and drops WER dramatically.

## Inference

```bash
python transcribe.py path/to/audio.wav
```

The script loads the fine-tuned checkpoint, transcribes a single audio file, and
prints the text. (Requires an audio file; `ffmpeg` is needed for non-WAV inputs.)

---

## License

- Code in this folder: Apache-2.0 (see root `LICENSE`).
- Base model: **MIT** (Whisper) — see
  [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny).
- Dataset: LibriSpeech (CC BY 4.0).
