# Music Annotation Assistant

Local-first AI-assisted pre-labeling for music audio. It analyzes a full mix and produces instrument candidates, approximate activity intervals, global stereo descriptors, and reviewable JSON/CSV output.

> This is a pre-labeling assistant, not a fully automatic annotation system. Predictions are intended to be reviewed by a human.

## Why this exists

The project started as a way to reduce repetitive manual work in music-audio annotation. Instead of listening to every track from scratch and filling every field by hand, it generates a conservative first pass that a reviewer can accept, edit, or reject.

The public repository intentionally contains **no private datasets, no private platform captures, and no model checkpoints trained on private data**. Public evaluation uses documented public datasets and upstream pretrained models.

## What it does

- Local decoding for WAV, MP3, FLAC, M4A, AAC, and OGG.
- Instrument candidate scoring from Essentia's Discogs EffNet embedding model and MTG-Jamendo instrument head.
- Canonical instrument mapping with parent/subtype ambiguity protection.
- Per-label score statistics and approximate activity intervals.
- Global stereo analysis: pan, width, correlation, stability, movement, and phase warnings.
- FastAPI + SQLite + browser UI.
- JSON/CSV export and append-only human corrections/review events.
- Optional Demucs separation provider for conservative stem-assisted analysis.
- Public-dataset evaluation and threshold calibration utilities.

## What it does not do

- It does not claim to identify every instrument in a dense mix.
- It does not automatically submit forms or control third-party annotation websites.
- It does not upload audio or send telemetry.
- It does not treat the global stereo image as per-instrument spatial truth.
- It does not ship private fine-tuned checkpoints.

## Quick start

Python 3.10-3.14 is supported.

```bash
make setup
make setup-model
make download-models
make verify-models
make test
make run
```

Open `http://127.0.0.1:8000`.

Command-line analysis:

```bash
.venv/bin/python scripts/analyze.py /path/to/audio.wav
```

Results are written under `outputs/<analysis_id>/`.

## Public evaluation

```bash
make setup-evaluation
make download-evaluation
make evaluate
```

The current public calibration profile was selected for conservative pre-labeling. On the expanded 998-clip public evaluation described in [`docs/EVALUATION.md`](docs/EVALUATION.md), confirmed predictions reached 93.7% precision at 13.1% coverage; confirmed + candidate coverage was 39.4%. These are engineering validation numbers, not a claim of complete instrumentation recognition.

## Optional source separation

```bash
make setup-separation
```

Separation is disabled by default. The Demucs integration is deliberately conservative: semantic stems such as drums/bass/vocals may support matching categories, while the broad `other` stem is not used to assert piano, guitar, or strings.

## Privacy and data boundary

Audio is processed locally. Runtime audio, downloaded models, evaluation media, stems, SQLite databases, generated results, and caches are ignored by Git.

This OSS version is kept separate from any private annotation corpus or private fine-tuned model. See [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md).

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The main design goal is to keep model providers replaceable and preserve uncertainty instead of forcing a label when evidence is weak.

## Model and dependency licenses

The repository code is MIT licensed. Some optional runtime dependencies and upstream model assets have their own licenses. In particular, `essentia-tensorflow` is AGPL-3.0-only. Review [`docs/MODEL_LICENSES.md`](docs/MODEL_LICENSES.md) before redistribution or commercial deployment.

## Contributing

Issues, reproducible evaluation cases, provider integrations, and improvements to conservative decision logic are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## 中文简介

这是一个完全本地运行的音乐音频智能预标注助手。输入一段完整混音后，它会给出乐器候选、可能的出现区间、整体立体声信息，并输出可人工修改的 JSON / CSV。它更强调“高置信候选 + 人工复核”，不追求把所有未知项强行自动填写。
