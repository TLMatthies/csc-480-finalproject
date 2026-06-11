from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from wordplay_detectors import (
    GROUP_SIZE,
    detect_hidden_word,
    detect_homophone,
    detect_letter_add_drop,
    detect_mutual_anagram,
    detect_one_letter_off,
    detect_scrambled_word,
    score_group,
)


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "Connections_Data.parquet"

# A group "fires" when at least three of its four words fit the pattern.
FIRE_THRESHOLD = 0.75

DETECTOR_BY_PATTERN = {
    "mutual anagram": detect_mutual_anagram,
    "scrambled word": detect_scrambled_word,
    "add a letter": detect_letter_add_drop,
    "drop a letter": detect_letter_add_drop,
    "one letter off": detect_one_letter_off,
    "hidden word": detect_hidden_word,
    "homophone": detect_homophone,
}

# Purple-style groups that should fire, with the pattern and minimum score
# each one is expected to produce.
POSITIVE_CASES = [
    ("mutual anagram", ["LISTEN", "SILENT", "ENLIST", "TINSEL"], 1.0),
    ("scrambled word", ["EARTH", "BELOW", "REACT", "ANGEL"], 1.0),
    ("drop a letter", ["BREAD", "SPILL", "TRACE", "CLOVER"], 1.0),
    ("add a letter", ["RIGHT", "OIL", "RAIN", "EACH"], 1.0),
    ("one letter off", ["HOUSE", "BEER", "GOAL", "WARM"], 1.0),
    ("hidden word", ["CHIPMUNK", "SPEARMINT", "CHARM", "SHINGLE"], 1.0),
    ("homophone", ["BEE", "SEA", "JAY", "CUE"], 1.0),
    ("homophone", ["WON", "ATE", "FOR", "TOO"], 1.0),
    # 3/4 partial match: the odd word out should drag the score to 0.75.
    ("hidden word", ["CHIPMUNK", "SPEARMINT", "CHARM", "BUCKET"], 0.75),
]

# Meaning-based groups from real puzzles that no form detector should fire on.
NEGATIVE_CASES = [
    ("wet weather", ["SNOW", "HAIL", "SLEET", "RAIN"]),
    ("nba teams", ["HEAT", "BUCKS", "JAZZ", "NETS"]),
    ("keyboard keys", ["SHIFT", "TAB", "RETURN", "OPTION"]),
    ("magazines", ["TIME", "PEOPLE", "ESSENCE", "US"]),
    ("footwear", ["BOOT", "SANDAL", "SLIPPER", "LOAFER"]),
    ("moods", ["ANGRY", "GRUMPY", "JOLLY", "MOODY"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the wordplay detector fixture tests."
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Also score every historical group and report fire rates by level.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the Connections parquet data. Default: {DEFAULT_DATA_PATH}",
    )
    return parser.parse_args()


def run_positive_cases() -> list[str]:
    failures = []
    for expected_pattern, words, min_score in POSITIVE_CASES:
        result = score_group(words)
        detector_result = DETECTOR_BY_PATTERN[expected_pattern](words)
        if result.pattern != expected_pattern or result.score < min_score:
            failures.append(
                f"expected {expected_pattern} >= {min_score} for {words}, "
                f"got: {result.explain()}"
            )
        elif detector_result.score < min_score:
            failures.append(
                f"{expected_pattern} detector alone scored {detector_result.score} "
                f"on {words}, expected >= {min_score}"
            )
        else:
            print(f"  pass: {result.explain()}")
    return failures


def run_negative_cases() -> list[str]:
    failures = []
    for label, words in NEGATIVE_CASES:
        result = score_group(words)
        if result.score >= FIRE_THRESHOLD:
            failures.append(f"{label} {words} should stay quiet, got: {result.explain()}")
        else:
            print(f"  pass: {label} stayed quiet (best was {result.explain()})")
    return failures


def load_groups(data_path: Path) -> list[tuple[int, list[str]]]:
    import pandas as pd

    data = pd.read_parquet(data_path)
    groups = []
    for _, rows in data.groupby(["Game ID", "Group Name"]):
        words = rows["Word"].astype(str).tolist()
        if len(words) == GROUP_SIZE:
            groups.append((int(rows["Group Level"].iloc[0]), words))
    return groups


def run_history(data_path: Path) -> None:
    groups = load_groups(data_path)
    partial_by_level: Counter[int] = Counter()
    full_by_level: Counter[int] = Counter()
    total_by_level: Counter[int] = Counter()
    examples: list[str] = []

    for level, words in groups:
        total_by_level[level] += 1
        result = score_group(words)
        if result.score >= FIRE_THRESHOLD:
            partial_by_level[level] += 1
        if result.score >= 1.0:
            full_by_level[level] += 1
            if level == 3 and len(examples) < 10:
                examples.append(result.explain())

    # Level 3 is the purple row. A signal that measures wordplay should fire
    # more often as the level climbs, and most of all on purple.
    print("\nFire rate on historical groups, by difficulty level:")
    print(f"  {'level':>7}  {'>= ' + str(FIRE_THRESHOLD) + ' (3/4)':>14}  {'== 1.0 (4/4)':>14}")
    for level in sorted(total_by_level):
        total = total_by_level[level]
        partial = partial_by_level[level]
        full = full_by_level[level]
        print(f"  {level:>7}  {partial:4d}/{total} ({partial / total:5.1%})  "
              f"{full:4d}/{total} ({full / total:5.1%})")

    full_total = sum(full_by_level.values())
    if full_total:
        purple_share = full_by_level[3] / full_total
        print(f"\n  Purple's share of full (4/4) fires: {full_by_level[3]}/{full_total} "
              f"({purple_share:.0%}) against a 25% base rate.")

    print("\nSample purple groups caught at full strength:")
    for example in examples:
        print(f"  {example}")


def main() -> None:
    args = parse_args()

    print("Positive cases (purple groups that should fire):")
    failures = run_positive_cases()
    print("\nNegative cases (semantic groups that should stay quiet):")
    failures += run_negative_cases()

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for failure in failures:
            print(f"  {failure}")
    else:
        print(f"\nAll {len(POSITIVE_CASES) + len(NEGATIVE_CASES)} fixture cases passed.")

    if args.history:
        run_history(args.data_path)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
