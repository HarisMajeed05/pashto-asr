# Pashto ASR Fine-Tuning

Fine-tunes Whisper on Pashto, one of hundreds of languages that remain
seriously underserved in speech AI despite 60 million speakers. Reports a
real before/after Word Error Rate, the honest headline is the improvement,
not a state-of-the-art claim.


## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # accept Common Voice's terms once, in a browser
```

## 1. Prepare the data

```bash
python scripts/prepare_data.py --out data/
```

## 2. Baseline, before fine-tuning

```bash
python scripts/run_eval.py --model openai/whisper-small --data data/ --max-samples 100
```

Run this first, on the stock model, before touching fine-tuning. This
number, how badly an off-the-shelf model does on Pashto today, is the
actual point being demonstrated, not something to skip past.

## 3. Fine-tune

```bash
python scripts/fine_tune.py --data data/ --base-model openai/whisper-small
```

GPU auto-detected, CPU fallback. Checkpoints every 200 steps, last 3 kept.
Stop any time with Ctrl+C, resume with:
```bash
python scripts/fine_tune.py --data data/ --resume checkpoints/pashto/checkpoint-<N>
```

## 4. After, compare against the baseline

```bash
python scripts/run_eval.py --model checkpoints/pashto/final --data data/ --max-samples 100
```

Report both numbers together. "WER dropped from X% to Y%" is a real,
honest, reportable result. A single post-fine-tuning number on its own
isn't, there's no baseline to know if it's good.

## Why Pashto specifically

~60 million speakers, a real population, not a niche case. Historically
near-zero platform support (Common Voice had no Pashto interface at all
before 2023), grown into a real community-built dataset since through
sustained volunteer effort, meaning there's now enough data to actually
train and measure something. Still meaningfully behind English/French-tier
languages in total hours, that gap is the real thing being measured here.

## Honest limitations

- **This will not reach English-tier accuracy.** No low-resource fine-tune
  does, that's not the goal. The relative improvement over the untouched
  base model is the real, useful, reportable result.
- **Common Voice's Pashto data quality varies** (community-recorded audio,
  varying microphones and accents), expect the WER to reflect that, not a
  clean lab-recorded number.
- **`whisper-small` is a reasonable starting point, not the only choice.**
  `whisper-base` trains faster if you're compute-constrained, `whisper-medium`
  if you have the GPU budget and want a stronger baseline to improve on.
