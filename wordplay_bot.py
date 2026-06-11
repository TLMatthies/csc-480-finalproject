from __future__ import annotations

from enum import Enum
from itertools import combinations

from bin_embeddings import _total_pairwise_cosine_distance
from guessed_constraints import PrevConstraint
from word_embeddings import SentenceTransformerEmbedder, TextEmbedder
from wordplay_detectors import (
    WordplayResult,
    detect_homophone,
    detect_mutual_anagram,
    detect_one_letter_off,
)

# A puzzle has 1820 possible groups of four, and the looser detectors (scrambled
# word, add/drop a letter, hidden word) match some coincidental subset almost
# every game -- measured under 1% precision as a group predictor. Only the tightly
# constrained detectors are safe to actually drive a guess, so the bot ranks groups
# with just these three. The full six-detector score_group is still the right call
# for explaining a group we already trust to the judge.
TRUSTED_DETECTORS = (detect_mutual_anagram, detect_one_letter_off, detect_homophone)

# How hard a form match pulls a group up the ranking: a qualifying match drops a
# group's effective distance by this much. Set to 0 for the plain embedding bot.
DEFAULT_FORM_WEIGHT = 1.0

# Only let a group's form score count once it is at least this strong. A full 4/4
# match from a trusted detector is reliable; 3/4 is mostly coincidence, so by
# default only full matches help.
DEFAULT_FORM_THRESHOLD = 1.0


def _best_trusted_form(words: list[str]) -> WordplayResult:
    return max((detector(words) for detector in TRUSTED_DETECTORS),
               key=lambda result: result.score)


class WordplayConnectionBot:
    class Correctness(Enum):
        FullyWrong = PrevConstraint.Correctness.FullyWrong
        OneOff = PrevConstraint.Correctness.OneOff
        Correct = PrevConstraint.Correctness.Correct

    def __init__(
        self,
        words: list[str],
        embedder: TextEmbedder | None = None,
        form_weight: float = DEFAULT_FORM_WEIGHT,
        form_threshold: float = DEFAULT_FORM_THRESHOLD,
    ) -> None:
        if len(words) != 16 or len(set(words)) != 16:
            raise ValueError("WordplayConnectionBot needs 16 unique words!")

        self.words = list(words)
        self.form_weight = form_weight
        self.form_threshold = form_threshold
        self.constraints = PrevConstraint(words)
        self.bad_guesses: list[frozenset[str]] = []
        self.last_form: WordplayResult | None = None

        if embedder is None:
            embedder = SentenceTransformerEmbedder()
        vectors = embedder.embed_texts(self.words)
        position = {word: index for index, word in enumerate(self.words)}

        # Embedding distance and form score don't change as the game goes on, so
        # score every group of four once here and just filter the survivors later.
        self.distance: dict[frozenset[str], float] = {}
        self.form: dict[frozenset[str], WordplayResult] = {}
        for group in combinations(self.words, 4):
            key = frozenset(group)
            indices = tuple(position[word] for word in group)
            self.distance[key] = _total_pairwise_cosine_distance(indices, vectors)
            self.form[key] = _best_trusted_form(list(group))

    def _combined_score(self, key: frozenset[str]) -> float:
        form = self.form[key].score
        bonus = form if form >= self.form_threshold else 0.0
        return self.distance[key] - self.form_weight * bonus

    def get_guess(self) -> list[str]:
        if not self.words:
            raise ValueError("WordplayConnectionBot has no words left to guess.")

        already_wrong = set(self.bad_guesses)
        candidates = [
            frozenset(group)
            for group in combinations(self.words, 4)
            if frozenset(group) not in already_wrong
        ]
        candidates.sort(key=self._combined_score)

        for key in candidates:
            guess = [word for word in self.words if word in key]
            if self.constraints.is_possible(guess[0], guess[1], guess[2], guess[3]):
                self.last_form = self.form[key]
                return guess
            self.bad_guesses.append(key)

        raise ValueError("Every remaining group has been ruled out.")

    def add_guess_result(self, guess: list[str], correctness: Correctness) -> None:
        self.constraints.add_guess(guess[0], guess[1], guess[2], guess[3], correctness.value)
        self.bad_guesses.append(frozenset(guess))
        if correctness == self.Correctness.Correct:
            solved = set(guess)
            self.words = [word for word in self.words if word not in solved]
