"""
scripts/fine_tune.py
=======================
Fine-tunes a Whisper checkpoint on Pashto Common Voice data. GPU
auto-detected with a CPU fallback, periodic checkpointing, safe resume
after an interrupted run, via Hugging Face's Seq2SeqTrainer.

USAGE
-----
    python scripts/fine_tune.py --data data/ --base-model openai/whisper-small
    python scripts/fine_tune.py --data data/ --resume checkpoints/pashto/checkpoint-500
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import load_from_disk, Audio
from transformers import (
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from common import load_processor, prepare_example, WhisperDataCollator, MAX_LABEL_LENGTH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="Path produced by prepare_data.py")
    ap.add_argument("--base-model", default="openai/whisper-small",
                     help="Starting checkpoint. whisper-small balances quality against training "
                          "time, whisper-base trains faster")
    ap.add_argument("--output-dir", default="checkpoints/pashto")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--checkpoint-interval", type=int, default=200, help="Steps between checkpoints")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--resume", default=None, help="Checkpoint folder to resume from")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("--device cuda requested but no GPU found, falling back to CPU.")
        use_fp16 = False
    elif args.device == "auto":
        use_fp16 = torch.cuda.is_available()
        print(f"GPU detected: {use_fp16}" if use_fp16 else "No GPU detected, training on CPU (will be slow).")
    else:
        use_fp16 = args.device == "cuda"

    processor = load_processor(args.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    dataset = load_from_disk(args.data)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    dataset = dataset.filter(lambda ex: ex["sentence"] is not None and ex["sentence"].strip() != "")
    dataset = dataset.map(
        lambda b: prepare_example(b, processor),
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )

    before = {split: len(dataset[split]) for split in dataset}
    dataset = dataset.filter(lambda ex: len(ex["labels"]) <= MAX_LABEL_LENGTH)
    for split in dataset:
        dropped = before[split] - len(dataset[split])
        if dropped:
            print(f"  Dropped {dropped} {split} examples with labels longer than {MAX_LABEL_LENGTH} tokens")

    data_collator = WhisperDataCollator(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=args.checkpoint_interval,
        save_strategy="steps",
        save_steps=args.checkpoint_interval,
        save_total_limit=3,  # keeps only the most recent 3 checkpoints
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("test")),
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print(f"Fine-tuning {args.base_model} on Pashto data, {len(dataset['train'])} training examples")
    print(f"Checkpoints saved to {args.output_dir} every {args.checkpoint_interval} steps, last 3 kept")

    try:
        trainer.train(resume_from_checkpoint=args.resume)
    except KeyboardInterrupt:
        print("\nStopped early. The most recent periodic checkpoint is already saved.")
        print(f"Resume with: python scripts/fine_tune.py --data {args.data} --resume {args.output_dir}/<latest-checkpoint-folder>")
        sys.exit(0)

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nDone. Final model saved to: {final_dir}")
    print(f"Compare against the baseline:")
    print(f"  python scripts/evaluate.py --model {args.base_model} --data {args.data}")
    print(f"  python scripts/evaluate.py --model {final_dir} --data {args.data}")


if __name__ == "__main__":
    main()