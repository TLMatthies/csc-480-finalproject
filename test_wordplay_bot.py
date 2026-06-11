from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from test_connection_bot import DEFAULT_DATA_PATH, Puzzle, load_puzzles, score_guess
from word_embeddings import SentenceTransformerEmbedder
from wordplay_bot import DEFAULT_FORM_WEIGHT, WordplayConnectionBot, _best_trusted_form

STARTING_MISTAKES = 4
Correctness = WordplayConnectionBot.Correctness


@dataclass
class Outcome:
    won: bool
    groups_solved: int
    mistakes: int
    form_guesses: int = 0          # guesses made on a group with a 4/4 form match
    form_guesses_correct: int = 0  # of those, how many were a real group


@dataclass
class Tally:
    wins: int = 0
    games: int = 0
    mistakes: int = 0
    groups_solved: int = 0

    def add(self, outcome: Outcome) -> None:
        self.games += 1
        self.wins += int(outcome.won)
        self.mistakes += outcome.mistakes
        self.groups_solved += outcome.groups_solved

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def avg_mistakes(self) -> float:
        return self.mistakes / self.games if self.games else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the embedding bot against the embedding + wordplay bot."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of puzzles to run. Default: 50.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the puzzle sample. Default: 0.")
    parser.add_argument("--form-weight", type=float, default=DEFAULT_FORM_WEIGHT,
                        help=f"Form weight for the wordplay bot. Default: {DEFAULT_FORM_WEIGHT}.")
    parser.add_argument("--wordplay-only", action="store_true",
                        help="Only run puzzles that contain a trusted-detector wordplay group.")
    return parser.parse_args()


def true_groups(puzzle: Puzzle) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for word in puzzle.words:
        groups.setdefault(puzzle.word_to_group[word], []).append(word)
    return list(groups.values())


def has_covered_wordplay(puzzle: Puzzle) -> bool:
    """True if one of the four real groups is a trusted wordplay pattern."""
    return any(_best_trusted_form(group).score >= 1.0 for group in true_groups(puzzle))


def play(puzzle: Puzzle, embedder, form_weight: float) -> Outcome:
    bot = WordplayConnectionBot(puzzle.words, embedder=embedder, form_weight=form_weight)
    solved: set[str] = set()
    mistakes = 0
    form_guesses = 0
    form_correct = 0

    while mistakes < STARTING_MISTAKES and len(solved) < 3:
        guess = bot.get_guess()
        used_form = bot.last_form is not None and bot.last_form.score >= bot.form_threshold
        correctness, group = score_guess(guess, puzzle.word_to_group, WordplayConnectionBot)
        bot.add_guess_result(guess, correctness)

        correct = correctness == Correctness.Correct
        if used_form:
            form_guesses += 1
            form_correct += int(correct)
        if correct and group is not None:
            solved.add(group)
        elif not correct:
            mistakes += 1

    return Outcome(len(solved) >= 3, len(solved), mistakes, form_guesses, form_correct)


def print_report(baseline: Tally, wordplay: Tally, wp_baseline: Tally, wp_wordplay: Tally,
                 form_guesses: int, form_correct: int) -> None:
    def column(name: str, base: Tally, form: Tally) -> None:
        print(f"\n{name} ({base.games} puzzles)")
        print(f"  {'':<20}{'baseline':>12}{'+ wordplay':>12}")
        print(f"  {'wins':<20}{base.wins:>12}{form.wins:>12}")
        print(f"  {'win rate':<20}{base.win_rate:>11.1%}{form.win_rate:>12.1%}")
        print(f"  {'avg mistakes':<20}{base.avg_mistakes:>12.2f}{form.avg_mistakes:>12.2f}")
        print(f"  {'groups solved':<20}{base.groups_solved:>12}{form.groups_solved:>12}")

    print("\nEmbedding bot vs. embedding + wordplay bot")
    print("==========================================")
    column("All puzzles", baseline, wordplay)
    column("Puzzles with a detectable wordplay group", wp_baseline, wp_wordplay)

    precision = form_correct / form_guesses if form_guesses else 0.0
    print("\nForm-driven guesses (the bot guessed a group on a 4/4 form match):")
    print(f"  {form_correct}/{form_guesses} were a real group ({precision:.0%} precision)")


def main() -> None:
    args = parse_args()
    puzzles = load_puzzles(args.data_path)
    if args.wordplay_only:
        puzzles = [puzzle for puzzle in puzzles if has_covered_wordplay(puzzle)]
        print(f"Found {len(puzzles)} puzzles with a trusted-detector wordplay group.")
    else:
        random.seed(args.seed)
        random.shuffle(puzzles)
        puzzles = puzzles[: args.limit]

    embedder = SentenceTransformerEmbedder()
    embedder.embed_texts(["warmup"])  # load the model once before timing games

    baseline = Tally()
    wordplay = Tally()
    wp_baseline = Tally()
    wp_wordplay = Tally()
    form_guesses = 0
    form_correct = 0

    for index, puzzle in enumerate(puzzles, start=1):
        base_outcome = play(puzzle, embedder, form_weight=0.0)
        form_outcome = play(puzzle, embedder, form_weight=args.form_weight)
        baseline.add(base_outcome)
        wordplay.add(form_outcome)
        form_guesses += form_outcome.form_guesses
        form_correct += form_outcome.form_guesses_correct

        if has_covered_wordplay(puzzle):
            wp_baseline.add(base_outcome)
            wp_wordplay.add(form_outcome)

        if index % 10 == 0:
            print(f"  played {index}/{len(puzzles)} puzzles...")

    print_report(baseline, wordplay, wp_baseline, wp_wordplay, form_guesses, form_correct)


if __name__ == "__main__":
    main()
