"""Fine-tune a pretrained transformer for sentence generation on ISL vocabulary."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from src.utils import load_config, load_model_config, resolve_path, seed_everything

LOGGER = logging.getLogger(__name__)


class VocabularyDataset(Dataset):
    def __init__(self, sentences: List[str], tokenizer, max_length: int) -> None:
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int):
        text = self.sentences[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_sentences_from_vocab(vocab_df: pd.DataFrame) -> List[str]:
    sentences: List[str] = []
    for word in vocab_df["word"].dropna().astype(str):
        sentences.append(word.strip())
        sentences.append(f"This sign means {word.strip().lower()} in Indian Sign Language.")
    return sentences


def fine_tune() -> None:
    config = load_config()
    model_config = load_model_config()["transformer"]
    seed_everything(config["project"]["seed"])

    checkpoint_dir = resolve_path(config["paths"]["transformer_checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vocab_file = resolve_path(config["training"]["transformer"]["vocab_file"])
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}")

    vocab_df = pd.read_csv(vocab_file)
    sentences = build_sentences_from_vocab(vocab_df)

    pretrained_model = model_config.get("pretrained_model", "gpt2")
    transformer_cfg = config["training"]["transformer"]
    continue_from_checkpoint = bool(transformer_cfg.get("continue_from_checkpoint", False))
    additional_epochs = int(transformer_cfg.get("additional_epochs", 0))

    checkpoint_has_state = checkpoint_dir.exists() and any(checkpoint_dir.iterdir())
    tokenizer_source = checkpoint_dir if continue_from_checkpoint and checkpoint_has_state else pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = VocabularyDataset(sentences, tokenizer, model_config.get("max_seq_length", 16))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model_source = checkpoint_dir if continue_from_checkpoint and checkpoint_has_state else pretrained_model
    model = AutoModelForCausalLM.from_pretrained(model_source)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["optimizer"]["lr"])
    num_epochs = int(transformer_cfg["num_epochs"])
    total_epochs = max(num_epochs + additional_epochs, num_epochs)
    total_steps = total_epochs * len(dataloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(total_epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        LOGGER.info("Epoch %d/%d - Loss: %.4f", epoch + 1, total_epochs, avg_loss)

    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    LOGGER.info("Transformer fine-tuning complete. Checkpoint saved to %s", checkpoint_dir)


if __name__ == "__main__":
    fine_tune()
