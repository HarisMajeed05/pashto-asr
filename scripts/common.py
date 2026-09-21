"""
scripts/common.py
====================
Shared preprocessing pipeline: converts raw Common Voice audio+text into
Whisper's expected input format (log-mel spectrogram features + tokenized
labels), plus the data collator that batches variable-length examples for
training.
"""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import WhisperProcessor


def load_processor(model_id: str, language: str = "pashto"):
    return WhisperProcessor.from_pretrained(model_id, language=language, task="transcribe")


def prepare_example(batch, processor: WhisperProcessor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
