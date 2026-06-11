# Form-based wordplay detectors for the purple Connections category.
#
# The embedding and constraint signals reason about what words mean, so they
# cannot see categories built on spelling or sound (anagrams, hidden words,
# homophones). Each detector here checks one mechanical pattern over a group
# of four words and reports the fraction that pass along with per-word
# evidence, so the rest of the solver can see why a pattern fired.
#
# Known limitations:
#   - The themed detectors (one letter off, hidden word, homophone) can only
#     find categories whose theme appears in THEME_SETS below.
#   - Scramble targets and letter-edit results are gated by word frequency,
#     so a pattern built on an obscure word will be missed.
#   - Homophone coverage is limited to what the CMU dictionary knows.
#   - Theme words made of common letter runs (red, ant, rat) can show up
#     inside unrelated words and cause hidden-word false hits.

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import re
import string

GROUP_SIZE = 4
MIN_SCRAMBLE_LENGTH = 4
MIN_EDIT_RESULT_LENGTH = 4
MIN_HIDDEN_LENGTH = 3
MIN_ZIPF_FREQUENCY = 3.2

# Edits at the end of a word that just inflect it (plurals, past tense, -y
# adjectives) say nothing about wordplay, so they never count as matches.
INFLECTION_LETTERS = frozenset("sdy")

# The web2 list contains slurs, and an anagram or scramble lookup will happily
# surface one as a "rearranges to ..." target in the evidence text. They are
# dropped from the dictionary so they can never appear in output. This list is
# deliberately small and is not meant to be an exhaustive profanity filter.
BLOCKED_WORDS = frozenset({
    "nigger", "niger", "negress", "coon", "spic", "spick", "chink", "gook",
    "kike", "wop", "dago", "wetback", "beaner", "kraut", "jap", "redskin",
    "gyp", "fag", "faggot", "dyke", "tranny", "retard", "retarded", "spaz",
    "cripple", "midget", "mongoloid",
})

_STRESS_RE = re.compile(r"\d")

# Themed detectors can only find categories listed here -- extend as needed.
THEME_SETS: dict[str, frozenset[str]] = {
    "body parts": frozenset({
        "arm", "ear", "eye", "hip", "jaw", "leg", "lip", "rib", "toe", "gum",
        "gut", "shin", "chin", "knee", "heel", "nose", "brow", "calf", "hand",
        "foot", "neck", "back", "spine", "thigh", "wrist", "ankle", "elbow",
        "thumb",
    }),
    "animals": frozenset({
        "ant", "ape", "asp", "bat", "bee", "boa", "cat", "cow", "doe", "dog",
        "eel", "elk", "emu", "ewe", "fox", "hen", "hog", "owl", "pig", "ram",
        "rat", "sow", "yak", "bear", "bird", "crab", "crow", "deer", "duck",
        "fish", "frog", "goat", "hare", "hawk", "lamb", "lion", "mole", "mule",
        "newt", "pony", "seal", "slug", "swan", "toad", "wasp", "wolf", "worm",
        "moose", "mouse", "sheep", "snake",
    }),
    "colors": frozenset({
        "red", "tan", "blue", "cyan", "gold", "gray", "grey", "jade", "pink",
        "plum", "rust", "teal", "amber", "beige", "black", "brown", "coral",
        "green", "ivory", "olive", "peach", "white", "indigo", "maroon",
        "orange", "purple", "violet", "yellow", "scarlet",
    }),
    "numbers": frozenset({
        "one", "two", "six", "ten", "zero", "four", "five", "nine", "three",
        "seven", "eight", "eleven", "twelve", "twenty", "thirty", "forty",
        "fifty", "sixty", "hundred", "thousand",
    }),
    "trees": frozenset({
        "oak", "elm", "fir", "ash", "yew", "pine", "palm", "aspen", "beech",
        "birch", "cedar", "maple", "spruce", "willow",
    }),
    "fruits": frozenset({
        "fig", "date", "kiwi", "lime", "pear", "plum", "apple", "grape",
        "lemon", "mango", "melon", "peach", "banana", "cherry",
    }),
    "metals": frozenset({
        "tin", "gold", "iron", "lead", "zinc", "brass", "steel", "bronze",
        "copper", "nickel", "silver",
    }),
}

LETTER_NAMES = frozenset(string.ascii_lowercase)


@dataclass(frozen=True)
class WordplayResult:
    pattern: str
    score: float
    matches: dict[str, str]
    theme: str | None = None

    def explain(self) -> str:
        if not self.matches:
            return f"{self.pattern}: 0/{GROUP_SIZE} words matched"
        details = "; ".join(
            f"{word} {detail}" for word, detail in sorted(self.matches.items())
        )
        theme_note = f" ({self.theme})" if self.theme else ""
        return (
            f"{self.pattern}{theme_note}: "
            f"{len(self.matches)}/{GROUP_SIZE} words matched - {details}"
        )


@lru_cache(maxsize=1)
def load_dictionary() -> frozenset[str]:
    """Load the web2 word list, keeping only reasonably common words."""
    from english_words import get_english_words_set
    from wordfreq import zipf_frequency

    web2 = get_english_words_set(["web2"], alpha=True, lower=True)
    return frozenset(
        word
        for word in web2
        if word not in BLOCKED_WORDS and zipf_frequency(word, "en") >= MIN_ZIPF_FREQUENCY
    )


@lru_cache(maxsize=1)
def load_signature_index() -> dict[str, frozenset[str]]:
    """Map each sorted-letters signature to the dictionary words that share it."""
    index: dict[str, set[str]] = {}
    for word in load_dictionary():
        index.setdefault("".join(sorted(word)), set()).add(word)
    return {signature: frozenset(words) for signature, words in index.items()}


@lru_cache(maxsize=1)
def load_homophone_targets() -> dict[str, list[tuple[str, str]]]:
    """Map stress-stripped CMU pronunciations to (theme, target word) pairs."""
    import pronouncing

    themes: dict[str, frozenset[str]] = {"letters": LETTER_NAMES, **THEME_SETS}
    targets: dict[str, list[tuple[str, str]]] = {}
    for theme, members in themes.items():
        for member in sorted(members):
            for phones in pronouncing.phones_for_word(member):
                targets.setdefault(_strip_stress(phones), []).append((theme, member))
    return targets


def _validate_group(words: list[str]) -> list[str]:
    if len(words) != GROUP_SIZE or len(set(words)) != GROUP_SIZE:
        raise ValueError(f"wordplay detectors need {GROUP_SIZE} unique words")
    return [word.strip().lower() for word in words]


def _strip_stress(phones: str) -> str:
    return _STRESS_RE.sub("", phones)


def _hamming_distance(left: str, right: str) -> int:
    return sum(1 for a, b in zip(left, right) if a != b)


def _position_label(index: int, last_index: int) -> str:
    if index == 0:
        return "first"
    if index == last_index:
        return "last"
    return "middle"


def _one_letter_edits(word: str) -> list[tuple[str, str, str, str]]:
    """List every (operation, position, letter, result) edit that makes a word."""
    dictionary = load_dictionary()
    edits = []
    for index, letter in enumerate(word):
        position = _position_label(index, len(word) - 1)
        if position == "last" and letter in INFLECTION_LETTERS:
            continue
        result = word[:index] + word[index + 1:]
        if len(result) >= MIN_EDIT_RESULT_LENGTH and result in dictionary:
            edits.append(("drop", position, letter, result))
    for index in range(len(word) + 1):
        position = _position_label(index, len(word))
        for letter in string.ascii_lowercase:
            if position == "last" and letter in INFLECTION_LETTERS:
                continue
            result = word[:index] + letter + word[index:]
            if len(result) >= MIN_EDIT_RESULT_LENGTH and result in dictionary:
                edits.append(("add", position, letter, result))
    return edits


def detect_mutual_anagram(words: list[str]) -> WordplayResult:
    """Score how many of the four words are rearrangements of each other."""
    normalized = _validate_group(words)
    signatures = {
        word: "".join(sorted(word)) for word in normalized if word.isalpha()
    }
    counts = Counter(signatures.values())
    if not counts or max(counts.values()) < 2:
        return WordplayResult("mutual anagram", 0.0, {})
    best_signature = counts.most_common(1)[0][0]
    matched = sorted(word for word, sig in signatures.items() if sig == best_signature)
    matches = {
        word: f"has the same letters as {', '.join(o for o in matched if o != word)}"
        for word in matched
    }
    return WordplayResult("mutual anagram", len(matched) / GROUP_SIZE, matches)


def detect_scrambled_word(words: list[str]) -> WordplayResult:
    """Score how many words rearrange into a different dictionary word."""
    normalized = _validate_group(words)
    index = load_signature_index()
    matches: dict[str, str] = {}
    for word in normalized:
        if not word.isalpha() or len(word) < MIN_SCRAMBLE_LENGTH:
            continue
        rearrangements = index.get("".join(sorted(word)), frozenset()) - {word}
        if rearrangements:
            examples = ", ".join(sorted(rearrangements)[:3])
            matches[word] = f"rearranges to {examples}"
    return WordplayResult("scrambled word", len(matches) / GROUP_SIZE, matches)


def detect_letter_add_drop(words: list[str]) -> WordplayResult:
    """Score how many words make a new word by adding or dropping one letter.

    Edits only count toward the score when they agree on a shared pattern
    (same operation and position, or same operation and letter), so a group
    where every word happens to shed some unrelated letter stays quiet.
    """
    normalized = _validate_group(words)
    grouped: dict[tuple[str, str, str], dict[str, str]] = {}
    for word in normalized:
        if not word.isalpha():
            continue
        for operation, position, letter, result in _one_letter_edits(word):
            sign = "+" if operation == "add" else "-"
            detail = f"{sign}{letter} -> {result}"
            keys = [(operation, "letter", letter)]
            if position != "middle":
                keys.append((operation, "position", position))
            for key in keys:
                grouped.setdefault(key, {}).setdefault(word, detail)
    if not grouped:
        return WordplayResult("add or drop a letter", 0.0, {})
    best_key, matches = max(grouped.items(), key=lambda item: (len(item[1]), item[0]))
    operation, kind, value = best_key
    if kind == "position":
        theme = f"{operation} the {value} letter"
    else:
        theme = f"{operation} the letter '{value}'"
    pattern = "add a letter" if operation == "add" else "drop a letter"
    return WordplayResult(pattern, len(matches) / GROUP_SIZE, matches, theme)


def detect_one_letter_off(words: list[str]) -> WordplayResult:
    """Score how many words are one substituted letter away from a themed word."""
    normalized = _validate_group(words)
    best_theme: str | None = None
    best_matches: dict[str, str] = {}
    for theme, members in THEME_SETS.items():
        matches: dict[str, str] = {}
        for word in normalized:
            if not word.isalpha() or word in members:
                continue
            targets = sorted(
                member
                for member in members
                if len(member) == len(word) and _hamming_distance(member, word) == 1
            )
            if targets:
                matches[word] = f"is one letter from {targets[0]}"
        if len(matches) > len(best_matches):
            best_theme, best_matches = theme, matches
    return WordplayResult(
        "one letter off", len(best_matches) / GROUP_SIZE, best_matches, best_theme
    )


def detect_hidden_word(words: list[str]) -> WordplayResult:
    """Score how many words contain a smaller themed word inside them."""
    normalized = _validate_group(words)
    best_theme: str | None = None
    best_matches: dict[str, str] = {}
    for theme, members in THEME_SETS.items():
        matches: dict[str, str] = {}
        for word in normalized:
            letters = "".join(ch for ch in word if ch.isalpha())
            found = [
                member
                for member in members
                if len(member) >= MIN_HIDDEN_LENGTH
                and len(letters) > len(member)
                and member in letters
            ]
            if found:
                matches[word] = f"contains {max(sorted(found), key=len)}"
        if len(matches) > len(best_matches):
            best_theme, best_matches = theme, matches
    return WordplayResult(
        "hidden word", len(best_matches) / GROUP_SIZE, best_matches, best_theme
    )


def detect_homophone(words: list[str]) -> WordplayResult:
    """Score how many words sound exactly like a themed word or letter name."""
    import pronouncing

    normalized = _validate_group(words)
    targets_by_phones = load_homophone_targets()
    theme_matches: dict[str, dict[str, str]] = {}
    for word in normalized:
        if not word.isalpha():
            continue
        for phones in pronouncing.phones_for_word(word):
            for theme, target in targets_by_phones.get(_strip_stress(phones), []):
                if target == word:
                    continue
                if theme == "letters":
                    detail = f"sounds like the letter {target.upper()}"
                else:
                    detail = f"sounds like {target}"
                theme_matches.setdefault(theme, {}).setdefault(word, detail)
    if not theme_matches:
        return WordplayResult("homophone", 0.0, {})
    best_theme, best_matches = max(
        theme_matches.items(), key=lambda item: (len(item[1]), item[0])
    )
    return WordplayResult(
        "homophone", len(best_matches) / GROUP_SIZE, best_matches, best_theme
    )


ALL_DETECTORS = [
    detect_mutual_anagram,
    detect_scrambled_word,
    detect_letter_add_drop,
    detect_one_letter_off,
    detect_hidden_word,
    detect_homophone,
]


def score_group(words: list[str]) -> WordplayResult:
    """Run every detector on a group of four and return the strongest match.

    Scores are the fraction of the group that fits the pattern, so 1.0 means
    all four words match and 0.75 is a weak signal that the odd word out may
    belong somewhere else. Earlier detectors in ALL_DETECTORS win ties, so
    tighter patterns (mutual anagram) beat looser ones at the same score.
    """
    return max(
        (detector(words) for detector in ALL_DETECTORS),
        key=lambda result: result.score,
    )
