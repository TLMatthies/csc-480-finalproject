from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np

from word_embeddings import DEFAULT_FASTTEXT_MODEL_PATH, FastTextBinEmbedder, TextEmbedder


DEFAULT_MODEL_PATH = DEFAULT_FASTTEXT_MODEL_PATH


def _total_pairwise_cosine_distance(indices: tuple[int, ...], vectors: np.ndarray) -> float:
    total_distance = 0.0
    for left, right in combinations(indices, 2):
        cosine_similarity = float(np.dot(vectors[left], vectors[right]))
        total_distance += 1.0 - cosine_similarity
    return total_distance


def _validate_words(words: list[str]) -> None:
    if len(words) == 0 or len(words) > 16 or len(words) % 4 != 0:
        raise ValueError("words must have length 4, 8, 12, or 16")
    if len(set(words)) != len(words):
        raise ValueError("words must be unique")


def _normalize_incorrect_guesses(
    incorrect_guesses: list[list[str]] | None,
) -> set[frozenset[str]]:
    if incorrect_guesses is None:
        return set()

    normalized = set()
    for guess in incorrect_guesses:
        if len(guess) != 4 or len(set(guess)) != 4:
            raise ValueError("each incorrect guess must contain four unique words")
        normalized.add(frozenset(guess))
    return normalized


def _vectors_for_words(
    words: list[str],
    model_path: str | Path,
    embedder: TextEmbedder | None,
) -> np.ndarray:
    if embedder is None:
        embedder = FastTextBinEmbedder(model_path)
    return embedder.embed_texts(words)


def make_best_connection_guess(
    words: list[str],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    embedder: TextEmbedder | None = None,
    incorrect_guesses: list[list[str]] | None = None,
) -> tuple[float, list[str]]:
    """
    Return the allowed group of four with the lowest total cosine distance.
    """
    _validate_words(words)
    forbidden_guesses = _normalize_incorrect_guesses(incorrect_guesses)
    vectors = _vectors_for_words(words, model_path, embedder)

    allowed_groups = [
        group
        for group in combinations(range(len(words)), 4)
        if frozenset(words[index] for index in group) not in forbidden_guesses
    ]
    if not allowed_groups:
        raise ValueError("incorrect_guesses excludes every possible group")

    best_group = min(
        allowed_groups,
        key=lambda group: _total_pairwise_cosine_distance(group, vectors),
    )
    return (
        _total_pairwise_cosine_distance(best_group, vectors),
        [words[index] for index in best_group],
    )


def make_connections_guesses(
    words: list[str],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    embedder: TextEmbedder | None = None,
    incorrect_guesses: list[list[str]] | None = None,
) -> list[tuple[float, list[str]]]:
    """
    Return guessed Connections groups of four using FastText cosine distance.

    The input must contain unique strings, have length 4, 8, 12, or 16, and use
    the same words as the returned guesses. Each returned tuple contains the
    group's total pairwise cosine distance and its member strings. Groups with
    lower distance are treated as more semantically cohesive.
    """
    _validate_words(words)
    forbidden_guesses = _normalize_incorrect_guesses(incorrect_guesses)
    vectors = _vectors_for_words(words, model_path, embedder)

    group_scores = {
        group: _total_pairwise_cosine_distance(group, vectors)
        for group in combinations(range(len(words)), 4)
        if frozenset(words[index] for index in group) not in forbidden_guesses
    }

    @lru_cache(maxsize=None)
    def best_partition(
        remaining: tuple[int, ...],
    ) -> tuple[float, tuple[tuple[int, ...], ...]] | None:
        if not remaining:
            return 0.0, ()

        first = remaining[0]
        candidates = []
        for rest in combinations(remaining[1:], 3):
            group = tuple(sorted((first, *rest)))
            if group not in group_scores:
                continue
            next_remaining = tuple(index for index in remaining if index not in group)
            next_partition = best_partition(next_remaining)
            if next_partition is None:
                continue
            next_score, next_groups = next_partition
            candidates.append((group_scores[group] + next_score, (group, *next_groups)))

        if not candidates:
            return None

        return min(candidates, key=lambda candidate: candidate[0])

    partition = best_partition(tuple(range(len(words))))
    if partition is None:
        raise ValueError("incorrect_guesses excludes every possible partition")

    _, groups = partition
    ranked_groups = sorted(groups, key=lambda group: group_scores[group])
    return [
        (group_scores[group], [words[index] for index in group])
        for group in ranked_groups
    ]
