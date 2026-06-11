from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "Connections_Data.parquet"
REQUIRED_COLUMNS = {"Game ID", "Word", "Group Name"}
STARTING_MISTAKES = 4


@dataclass
class Puzzle:
    game_id: int
    words: list[str]
    word_to_group: dict[str, str]


@dataclass
class Results:
    wins_by_mistakes_remaining: Counter[int] = field(default_factory=Counter)
    losses_by_groups_correct: Counter[int] = field(default_factory=Counter)
    completed_games: int = 0
    interrupted: bool = False

    def record_win(self, mistakes_remaining: int) -> None:
        self.wins_by_mistakes_remaining[mistakes_remaining] += 1
        self.completed_games += 1

    def record_loss(self, groups_correct: int) -> None:
        self.losses_by_groups_correct[groups_correct] += 1
        self.completed_games += 1

    @property
    def wins(self) -> int:
        return sum(self.wins_by_mistakes_remaining.values())

    @property
    def losses(self) -> int:
        return sum(self.losses_by_groups_correct.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ConnectionBot against the Connections parquet dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the Connections parquet data. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of games to test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible ConnectionBot choices.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress after this many completed games. Use 0 to disable.",
    )
    return parser.parse_args()


def load_puzzles(data_path: Path) -> list[Puzzle]:
    import pandas as pd

    try:
        data = pd.read_parquet(data_path)
    except ImportError as exc:
        raise SystemExit(
            "Unable to read parquet data. Install a parquet engine such as pyarrow "
            "or fastparquet, then run this script again."
        ) from exc

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{data_path} is missing required column(s): {missing}")

    puzzles: list[Puzzle] = []
    for game_id, game_rows in data.groupby("Game ID", sort=True):
        if len(game_rows) != 16:
            raise ValueError(f"Game {game_id} has {len(game_rows)} words instead of 16.")

        words = game_rows["Word"].astype(str).tolist()
        if len(set(words)) != 16:
            raise ValueError(f"Game {game_id} does not have 16 unique words.")

        group_sizes = game_rows.groupby("Group Name").size()
        bad_groups = group_sizes[group_sizes != 4]
        if not bad_groups.empty:
            raise ValueError(f"Game {game_id} has group(s) without exactly 4 words.")

        word_to_group = dict(
            zip(game_rows["Word"].astype(str), game_rows["Group Name"].astype(str))
        )
        puzzles.append(Puzzle(int(game_id), words, word_to_group))

    return puzzles


def score_guess(
    guess: list[str],
    word_to_group: dict[str, str],
    bot_class: Any,
) -> tuple[Any, str | None]:
    guessed_groups = [word_to_group[word] for word in guess]
    group_counts = Counter(guessed_groups)

    if len(group_counts) == 1:
        return bot_class.Correctness.Correct, guessed_groups[0]
    if max(group_counts.values()) == 3:
        return bot_class.Correctness.OneOff, None
    return bot_class.Correctness.FullyWrong, None


def run_puzzle(puzzle: Puzzle) -> tuple[bool, int]:
    from connection_bot import ConnectionBot

    bot = ConnectionBot(puzzle.words)
    solved_groups: set[str] = set()
    mistakes_remaining = STARTING_MISTAKES

    while mistakes_remaining > 0 and len(solved_groups) < 3:
        guess = bot.get_guess()
        correctness, solved_group = score_guess(guess, puzzle.word_to_group, ConnectionBot)
        bot.add_guess_result(guess, correctness)

        if correctness == ConnectionBot.Correctness.Correct:
            if solved_group is not None:
                solved_groups.add(solved_group)
        else:
            mistakes_remaining -= 1

    if len(solved_groups) >= 3:
        return True, mistakes_remaining
    return False, len(solved_groups)


def print_results(results: Results) -> None:
    print("\nConnectionBot test results")
    print("==========================")
    if results.interrupted:
        print("Interrupted by Ctrl+C; showing completed games only.")
    print(f"Games completed: {results.completed_games}")
    print(f"Correct games:   {results.wins}")
    print(f"Incorrect games: {results.losses}")

    print("\nCorrect games by mistakes remaining:")
    for mistakes_remaining in range(1, STARTING_MISTAKES + 1):
        count = results.wins_by_mistakes_remaining[mistakes_remaining]
        print(f"  {mistakes_remaining} mistake(s) remaining: {count}")

    print("\nIncorrect games by groups guessed correctly:")
    for groups_correct in range(3):
        count = results.losses_by_groups_correct[groups_correct]
        print(f"  {groups_correct} group(s) guessed correctly: {count}")


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    results = Results()
    try:
        puzzles = load_puzzles(args.data_path)
        if args.limit is not None:
            puzzles = puzzles[: args.limit]

        for puzzle in puzzles:
            won, score = run_puzzle(puzzle)
            if won:
                results.record_win(score)
            else:
                results.record_loss(score)

            if args.progress_every and results.completed_games % args.progress_every == 0:
                print(f"Completed {results.completed_games}/{len(puzzles)} games...")
    except KeyboardInterrupt:
        results.interrupted = True

    print_results(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        interrupted_results = Results(interrupted=True)
        print_results(interrupted_results)
