"""
scripts/prepare_data.py
==========================
Loads the Pashto subset of Mozilla Common Voice and preprocesses it into
the format Whisper expects: 16kHz audio arrays paired with tokenized
target text.

Pashto specifically: ~60 million speakers, historically near-zero platform
support (Mozilla's own Pashto interface didn't exist before 2023), grown
into a real community-built corpus since. Still far behind English/French
tier languages in total hours, that gap is the actual thing this project
measures.

USAGE
-----
    python scripts/prepare_data.py --out data/

Requires a Hugging Face account and accepting Common Voice's terms once at
huggingface.co/datasets/mozilla-foundation/common_voice_17_0, then:
    huggingface-cli login
before running this.
"""

import argparse
import os

from datasets import load_dataset, Audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="Where to save the processed dataset")
    ap.add_argument("--hf-dataset", default="mozilla-foundation/common_voice_17_0",
                     help="Which Common Voice release to use")
    ap.add_argument("--lang", default="ps", help="Language code, ps = Pashto")
    args = ap.parse_args()

    print(f"Loading {args.hf_dataset} ({args.lang})...")
    dataset = load_dataset(args.hf_dataset, args.lang)

    columns_to_keep = ["audio", "sentence"]
    dataset = dataset.select_columns(columns_to_keep)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    os.makedirs(args.out, exist_ok=True)
    dataset.save_to_disk(args.out)

    for split in dataset:
        print(f"{split}: {len(dataset[split])} examples")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
