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

IMPORTANT, as of this writing: Mozilla moved Common Voice off Hugging Face
entirely in October 2025, to a new platform called Mozilla Data Collective.
The old huggingface.co/datasets/mozilla-foundation/common_voice_* repos are
now empty shells, load_dataset() on them fails with EmptyDatasetError, not
an auth problem, the data genuinely isn't there anymore.

SETUP, one-time:
    1. Create an account at https://datacollective.mozillafoundation.org
    2. Find the Pashto Common Voice dataset there, copy its dataset ID
       from the end of that page's URL
    3. Get an API key from your account dashboard
    4. pip install "datacollective[hf]"   (the [hf] extra is needed for
       return_format="hf" below, without it you only get pandas back)
    5. export MDC_API_KEY=your-api-key-here   (or put it in a .env file)

USAGE
-----
    python scripts/prepare_data.py --dataset-id <the-id-from-step-2> --out data/

API NOTE: this was corrected after the first version guessed a client-class
API (`DataCollective(...).load_dataset(...)`) that doesn't match what's
actually installed. The real package exposes plain functions, verified by
inspecting the installed package directly:
    datacollective.load_dataset(dataset_id, return_format="hf" | "pandas")
returning a HuggingFace Dataset/DatasetDict directly when return_format="hf"
is available, or a pandas DataFrame otherwise, this script handles both.

This project's sandbox still can't reach datacollective.mozillafoundation.org
itself, so the exact column names THIS SPECIFIC Pashto dataset uses
couldn't be confirmed here. This script checks several likely names and
PRINTS which one it actually found, if it guesses wrong, the printed
column list tells you immediately what to pass via --audio-column /
--text-column instead of failing silently.
"""

import argparse
import os

from datasets import Dataset, DatasetDict, Audio

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file in the current directory into os.environ, if one exists
except ImportError:
    pass  # python-dotenv not installed, .env files are simply ignored, MDC_API_KEY must be set directly instead


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
        f"Could not guess the right column automatically. Actual columns found: {list(columns)}\n"
        f"Pass the correct one explicitly with --audio-column or --text-column."
    )


def normalize_split(hf_or_df, audio_override, text_override):
    """Accepts either a pandas DataFrame or an already-HF Dataset for one
    split, returns a proper HF Dataset with 'audio' (cast to 16kHz) and
    'sentence' columns, regardless of what the source called them."""
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

    return ds.cast_column("audio", Audio(sampling_rate=16000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", required=True,
                     help="The Pashto Common Voice dataset ID from its Mozilla Data Collective page URL")
    ap.add_argument("--out", default="data", help="Where to save the processed dataset")
    ap.add_argument("--audio-column", default=None, help="Override if auto-detection guesses wrong")
    ap.add_argument("--text-column", default=None, help="Override if auto-detection guesses wrong")
    args = ap.parse_args()

    import datacollective  # imported here so --help works without the package installed

    print(f"Loading dataset {args.dataset_id} from Mozilla Data Collective...")
    try:
        result = datacollective.load_dataset(args.dataset_id, return_format="hf")
    except ImportError:
        print('The "hf" extra isn\'t installed (pip install "datacollective[hf]"), falling back to pandas.')
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
        # A single Dataset or a single pandas DataFrame, no separate splits
        # reported, wrap it as "train" so the rest of the pipeline (which
        # expects a DatasetDict) works unchanged, split it yourself with
        # dataset["train"].train_test_split(test_size=0.1) if you want a
        # real held-out test set instead of evaluating on training data.
        print("No separate splits reported, treating the whole thing as 'train'.")
        print("Consider splitting it yourself (see the comment in this script) before fine-tuning.")
        dataset = DatasetDict({"train": normalize_split(result, args.audio_column, args.text_column)})

    dataset.save_to_disk(args.out)

    for split in dataset:
        print(f"{split}: {len(dataset[split])} examples")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()