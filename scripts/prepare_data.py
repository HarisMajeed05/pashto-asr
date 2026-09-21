"""
scripts/prepare_data.py
==========================
Loads Pashto Common Voice data and preprocesses it into the format Whisper
expects: 16kHz audio arrays paired with tokenized target text.

Pashto: ~60 million speakers, historically near-zero platform support,
grown into a real community-built corpus in recent years. Still far behind
English/French tier languages in total hours.

Mozilla moved Common Voice off Hugging Face in October 2025, to a platform
called Mozilla Data Collective. The old huggingface.co/datasets/
mozilla-foundation/common_voice_* repos are now empty shells.

SETUP:
    1. Account at https://datacollective.mozillafoundation.org
    2. Dataset ID from the end of the dataset's page URL
    3. API key from the account dashboard
    4. pip install "datacollective[hf]"
    5. MDC_API_KEY set in the environment or a .env file

USAGE
-----
    python scripts/prepare_data.py --dataset-id <id> --out data/

The installed datacollective package exposes plain functions, not a
client class:
    datacollective.load_dataset(dataset_id, return_format="hf" | "pandas")
Returns a HuggingFace Dataset/DatasetDict when "hf" works, a pandas
DataFrame otherwise. Both are handled below.

Column names vary by dataset. This script checks several likely names and
prints which one it finds, so a wrong guess is visible immediately instead
of failing silently.
"""

import argparse
import os

from datasets import Dataset, DatasetDict, Audio

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads MDC_API_KEY from a .env file into the environment, if one exists
except ImportError:
    pass  # python-dotenv not installed, .env files are ignored, MDC_API_KEY must be set directly


LIKELY_AUDIO_COLUMNS = ["audio", "path", "audio_path", "clip_path", "file"]
LIKELY_TEXT_COLUMNS = ["sentence", "text", "transcription", "transcript"]


def guess_column(columns, candidates, override):
    if override:
        if override not in columns:
            raise SystemExit(f"--override column '{override}' not found. Actual columns: {list(columns)}")
        return override
    for c in candidates:
        if c in columns:
            return c
    raise SystemExit(
        f"Could not match a column automatically. Actual columns found: {list(columns)}\n"
        f"Pass the correct one explicitly with --audio-column or --text-column."
    )


def normalize_split(hf_or_df, audio_override, text_override):
    """Accepts a pandas DataFrame or an HF Dataset for one split, returns
    a Dataset with standardized 'audio' (16kHz) and 'sentence' columns."""
    if hasattr(hf_or_df, "column_names"):  # already a datasets.Dataset
        columns = hf_or_df.column_names
        audio_col = guess_column(columns, LIKELY_AUDIO_COLUMNS, audio_override)
        text_col = guess_column(columns, LIKELY_TEXT_COLUMNS, text_override)
        print(f"  Using '{audio_col}' as audio, '{text_col}' as transcription")
        ds = hf_or_df.select_columns([audio_col, text_col])
        ds = ds.rename_columns({audio_col: "audio", text_col: "sentence"})
    else:  # pandas DataFrame
        columns = hf_or_df.columns
        audio_col = guess_column(columns, LIKELY_AUDIO_COLUMNS, audio_override)
        text_col = guess_column(columns, LIKELY_TEXT_COLUMNS, text_override)
        print(f"  Using '{audio_col}' as audio, '{text_col}' as transcription")
        ds = Dataset.from_pandas(hf_or_df[[audio_col, text_col]].rename(
            columns={audio_col: "audio", text_col: "sentence"}
        ))

    # A large_string-backed column cannot be cast directly to Audio
    # (ArrowNotImplementedError). Rebuilding via from_dict normalizes the
    # underlying Arrow type to plain string, which the Audio cast accepts.
    ds = Dataset.from_dict({col: ds[col] for col in ds.column_names})
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    before = len(ds)
    ds = ds.filter(lambda ex: ex["sentence"] is not None and ex["sentence"].strip() != "")
    dropped = before - len(ds)
    if dropped:
        print(f"  Dropped {dropped} examples with empty/missing transcription")
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", required=True,
                     help="Pashto Common Voice dataset ID, from its Mozilla Data Collective page URL")
    ap.add_argument("--out", default="data", help="Output path for the processed dataset")
    ap.add_argument("--audio-column", default=None, help="Overrides auto-detection")
    ap.add_argument("--text-column", default=None, help="Overrides auto-detection")
    ap.add_argument("--test-size", type=float, default=0.1,
                     help="Fraction held out as a test split when the source dataset has none")
    args = ap.parse_args()

    import datacollective  # imported here so --help works without the package installed

    print(f"Loading dataset {args.dataset_id} from Mozilla Data Collective...")
    try:
        result = datacollective.load_dataset(args.dataset_id, return_format="hf")
    except ImportError:
        print('The "hf" extra is not installed (pip install "datacollective[hf]"), falling back to pandas.')
        result = datacollective.load_dataset(args.dataset_id, return_format="pandas")

    os.makedirs(args.out, exist_ok=True)

    if isinstance(result, DatasetDict) or (hasattr(result, "keys") and not hasattr(result, "columns")):
        print(f"Splits reported: {list(result.keys())}")
        dataset_dict = {}
        for split_name, split_data in result.items():
            print(f"\n[{split_name}]")
            dataset_dict[split_name] = normalize_split(split_data, args.audio_column, args.text_column)
        dataset = DatasetDict(dataset_dict)
    else:
        # No splits reported, so a train/test split is cut here directly
        # (90/10 by default) rather than left for evaluate.py to fail on.
        print(f"No separate splits reported, cutting a {int((1 - args.test_size) * 100)}/{int(args.test_size * 100)} train/test split.")
        full_dataset = normalize_split(result, args.audio_column, args.text_column)
        dataset = full_dataset.train_test_split(test_size=args.test_size, seed=42)

    dataset.save_to_disk(args.out)

    for split in dataset:
        print(f"{split}: {len(dataset[split])} examples")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()