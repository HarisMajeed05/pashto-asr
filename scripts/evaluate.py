"""
scripts/evaluate.py
======================
Computes Word Error Rate (WER) on the Pashto test split. Run this TWICE:
once with the stock model (before), once with your fine-tuned checkpoint
(after). The gap between those two numbers is the actual result this
project is measuring, not a single absolute score.

USAGE
-----
    # Baseline, stock model, never seen Pashto fine-tuning
    python scripts/evaluate.py --model openai/whisper-small --data data/

    # After fine-tuning
    python scripts/evaluate.py --model checkpoints/pashto/final --data data/
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import evaluate as hf_evaluate
import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration

from common import load_processor, prepare_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model id or local checkpoint path")
    ap.add_argument("--data", default="data", help="Path from prepare_data.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=None, help="Limit for a quick sanity check")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.device == "auto" else args.device
    )

    processor = load_processor(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    dataset = load_from_disk(args.data)[args.split]
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    dataset = dataset.map(lambda b: prepare_example(b, processor), remove_columns=dataset.column_names)

    wer_metric = hf_evaluate.load("wer")
    predictions, references = [], []

    print(f"Evaluating {args.model} on {len(dataset)} Pashto test examples ({device})...")
    with torch.no_grad():
        for i, example in enumerate(dataset):
            input_features = torch.tensor(example["input_features"]).unsqueeze(0).to(device)
            predicted_ids = model.generate(input_features, language="pashto", task="transcribe")
            prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            labels = [t if t != -100 else processor.tokenizer.pad_token_id for t in example["labels"]]
            reference = processor.tokenizer.decode(labels, skip_special_tokens=True)

            predictions.append(prediction)
            references.append(reference)

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(dataset)}...")

    wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"\nModel: {args.model}")
    print(f"WER on {len(dataset)} Pashto test examples: {wer * 100:.2f}%")
    print("\nSample predictions vs references:")
    for p, r in list(zip(predictions, references))[:5]:
        print(f"  pred: {p}")
        print(f"  ref:  {r}")
        print()


if __name__ == "__main__":
    main()
