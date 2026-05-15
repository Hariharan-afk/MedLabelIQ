from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from medlabeliq.config.settings import settings


def get_device() -> torch.device:
    """
    Use GPU when available, otherwise fall back to CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def get_tokenizer():
    """
    Load the reranker tokenizer once per process.
    """
    return AutoTokenizer.from_pretrained(settings.reranker_model_name)


@lru_cache(maxsize=1)
def get_reranker_model():
    """
    Load the reranker model once per process.

    BGE reranker is a cross-encoder sequence-classification model that
    outputs a relevance logit for each (query, passage) pair.
    """
    device = get_device()

    model = AutoModelForSequenceClassification.from_pretrained(
        settings.reranker_model_name
    )
    model.to(device)
    model.eval()

    return model


def rerank_pairs(
    query: str,
    passages: list[str],
    *,
    batch_size: int = 8,
    max_length: int = 512,
) -> list[float]:
    """
    Score (query, passage) pairs using BAAI/bge-reranker-base.

    Returns raw relevance logits. Higher score = more relevant.
    Raw scores are sufficient because we only use them for ranking.
    """
    if not passages:
        return []

    tokenizer = get_tokenizer()
    model = get_reranker_model()
    device = get_device()

    all_scores: list[float] = []

    pairs = [[query, passage] for passage in passages]

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]

        inputs = tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            logits = model(
                **inputs,
                return_dict=True,
            ).logits.view(-1).float()

        all_scores.extend(logits.detach().cpu().tolist())

    return [float(score) for score in all_scores]