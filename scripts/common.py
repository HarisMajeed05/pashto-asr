"""
scripts/common.py
====================
Shared preprocessing pipeline: converts raw Common Voice audio+text into
Whisper's expected input format (log-mel spectrogram features + tokenized
labels), plus the data collator used to batch variable-length examples
for training.
"""

import io
from dataclasses import dataclass
from typing import Any

import librosa
import soundfile as sf
import torch
from transformers import WhisperProcessor

TARGET_SAMPLING_RATE = 16000


def load_processor(model_id: str, language: str = "pashto"):
    return WhisperProcessor.from_pretrained(model_id, language=language, task="transcribe")


def prepare_example(batch, processor: WhisperProcessor):
    audio = batch["audio"]
    if audio.get("bytes") is not None:
        array, sampling_rate = sf.read(io.BytesIO(audio["bytes"]))
    else:
        array, sampling_rate = sf.read(audio["path"])
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sampling_rate != TARGET_SAMPLING_RATE:
        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=TARGET_SAMPLING_RATE)
        sampling_rate = TARGET_SAMPLING_RATE
    batch["input_features"] = processor.feature_extractor(
        array, sampling_rate=sampling_rate
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
        # Strips the BOS token when the tokenizer already prepended one,
        # avoiding a double-BOS during training.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch