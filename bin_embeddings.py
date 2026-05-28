from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path
import re

import numpy as np


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "data" / "cc.en.300.bin"
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@lru_cache(maxsize=1)
def load_fasttext_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load and cache the FastText vectors from the binary model file."""
    from gensim.models.fasttext import load_facebook_vectors

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"FastText model not found: {path}")
    return load_facebook_vectors(str(path))


def _phrase_vector(text: str, model) -> np.ndarray:
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        tokens = [text.lower()]

    vectors = np.array([model.get_vector(token) for token in tokens], dtype=np.float32)
    vector = vectors.mean(axis=0)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(f"FastText returned a zero vector for {text!r}")
    return vector / norm


def _total_pairwise_cosine_distance(indices: tuple[int, ...], vectors: np.ndarray) -> float:
    total_distance = 0.0
    for left, right in combinations(indices, 2):
        cosine_similarity = float(np.dot(vectors[left], vectors[right]))
        total_distance += 1.0 - cosine_similarity
    return total_distance


def make_connections_guesses(
    words: list[str],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> list[tuple[float, list[str]]]:
    """
    Return guessed Connections groups of four using FastText cosine distance.

    The input must contain unique strings, have length 4, 8, 12, or 16, and use
    the same words as the returned guesses. Each returned tuple contains the
    group's total pairwise cosine distance and its member strings. Groups with
    lower distance are treated as more semantically cohesive.
    """
    if len(words) == 0 or len(words) > 16 or len(words) % 4 != 0:
        raise ValueError("words must have length 4, 8, 12, or 16")
    if len(set(words)) != len(words):
        raise ValueError("words must be unique")

    model = load_fasttext_model(model_path)
    vectors = np.array([_phrase_vector(word, model) for word in words], dtype=np.float32)

    group_scores = {
        group: _total_pairwise_cosine_distance(group, vectors)
        for group in combinations(range(len(words)), 4)
    }

    @lru_cache(maxsize=None)
    def best_partition(remaining: tuple[int, ...]) -> tuple[float, tuple[tuple[int, ...], ...]]:
        if not remaining:
            return 0.0, ()

        first = remaining[0]
        candidates = []
        for rest in combinations(remaining[1:], 3):
            group = tuple(sorted((first, *rest)))
            next_remaining = tuple(index for index in remaining if index not in group)
            next_score, next_groups = best_partition(next_remaining)
            candidates.append((group_scores[group] + next_score, (group, *next_groups)))

        return min(candidates, key=lambda candidate: candidate[0])

    _, groups = best_partition(tuple(range(len(words))))
    ranked_groups = sorted(groups, key=lambda group: group_scores[group])
    return [
        (group_scores[group], [words[index] for index in group])
        for group in ranked_groups
    ]
