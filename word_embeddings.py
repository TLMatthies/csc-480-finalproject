from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol
import logging
import os
import re

import numpy as np


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

DEFAULT_FASTTEXT_MODEL_PATH = Path(__file__).resolve().parent / "data" / "cc.en.300.bin"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
TEXT_ALIASES = {
    "🍞": "bread",
    "🥬": "lettuce",
    "🧀": "cheese",
    "🥓": "bacon",
    "🧠": "brain",
    "🌧️": "rain",
    "🚂": "train",
    "✈️": "plane",
    "🫖": "tea",
    "🐑": "ewe",
    "👁️": "eye",
    "🐝": "bee",
    "🪚": "saw",
    "😱": "scream",
    "👽": "alien",
    "🧛": "vampire",
}


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return one normalized vector per input string."""


def _normalize_vector(vector: np.ndarray, label: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(f"Embedding model returned a zero vector for {label!r}")
    return vector / norm


@lru_cache(maxsize=1)
def load_fasttext_model(model_path: str | Path = DEFAULT_FASTTEXT_MODEL_PATH):
    """Load and cache a FastText binary model."""
    from gensim.models.fasttext import load_facebook_vectors

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"FastText model not found: {path}")
    return load_facebook_vectors(str(path))


class FastTextBinEmbedder:
    def __init__(self, model_path: str | Path = DEFAULT_FASTTEXT_MODEL_PATH) -> None:
        self.model_path = model_path

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        model = load_fasttext_model(self.model_path)
        return np.array([self._embed_text(text, model) for text in texts], dtype=np.float32)

    def _embed_text(self, text: str, model) -> np.ndarray:
        embedding_text = TEXT_ALIASES.get(text, text)
        tokens = TOKEN_RE.findall(embedding_text.lower())
        if not tokens:
            tokens = [embedding_text.lower()]

        vectors = np.array([model.get_vector(token) for token in tokens], dtype=np.float32)
        vector = vectors.mean(axis=0)
        return _normalize_vector(vector, text)


def _quiet_sentence_transformer_logging() -> None:
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except ImportError:
        pass

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except ImportError:
        pass


@lru_cache(maxsize=2)
def load_sentence_transformer(model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL):
    """Load and cache a Sentence Transformers model."""
    _quiet_sentence_transformer_logging()

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL) -> None:
        self.model_name = model_name

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        model = load_sentence_transformer(self.model_name)
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vectors, dtype=np.float32)
