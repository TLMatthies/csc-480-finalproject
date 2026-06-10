from bin_embeddings import make_connections_guesses, make_best_connection_guess
from guessed_constraints import PrevConstraint
import random
from enum import Enum


class ConnectionBot:
    class Correctness(Enum):
        FullyWrong = PrevConstraint.Correctness.FullyWrong
        OneOff = PrevConstraint.Correctness.OneOff
        Correct = PrevConstraint.Correctness.Correct
        
    def __init__(self, words: list[str]) -> None:
        if len(words) != 16 or len(set(words)) != 16:
            raise ValueError("ConnectionBot must be initialized with a list of unique words 16 long!")
        self.constraints = PrevConstraint(words)
        self.bad_guesses: list[list[str]] = []
        self.words = words

    def get_guess(self)->list[str]:
        current_guess: list[str] = []
        while True:
            single_guess = make_best_connection_guess(self.words, 
                                                      incorrect_guesses=self.bad_guesses)
            multi_guesses = make_connections_guesses(self.words,
                                                     incorrect_guesses=self.bad_guesses)
            current_guess = random.choices(population=[single_guess[1], multi_guesses[0][1]],
                                           weights=[1/single_guess[0], 1/multi_guesses[0][0]],
                                           k=1)[0]

            if self.constraints.is_possible(current_guess[0], 
                                            current_guess[1], 
                                            current_guess[2], 
                                            current_guess[3]):
                break
            else:
                self.bad_guesses.append(current_guess)

        return current_guess

    def add_guess_result(self, guess: list[str], correctness: Correctness):
        self.constraints.add_guess(guess[0], guess[1], guess[2], guess[3], correctness.value)
        if correctness == self.Correctness.Correct:
            self.bad_guesses.append(guess)


def _read_words() -> list[str]:
    while True:
        raw_words = input("Enter the 16 words, separated by commas or spaces:\n> ").strip()
        words = [word.strip() for word in raw_words.replace(",", " ").split()]

        if len(words) != 16:
            print(f"Expected 16 words, got {len(words)}. Please try again.")
        elif len(set(words)) != 16:
            print("Expected 16 unique words. Please try again.")
        else:
            return words


def _read_correctness() -> ConnectionBot.Correctness:
    choices = {
        "c": ConnectionBot.Correctness.Correct,
        "correct": ConnectionBot.Correctness.Correct,
        "o": ConnectionBot.Correctness.OneOff,
        "one": ConnectionBot.Correctness.OneOff,
        "one off": ConnectionBot.Correctness.OneOff,
        "one-off": ConnectionBot.Correctness.OneOff,
        "i": ConnectionBot.Correctness.FullyWrong,
        "incorrect": ConnectionBot.Correctness.FullyWrong,
        "wrong": ConnectionBot.Correctness.FullyWrong,
    }

    while True:
        raw_result = input("Was this guess correct, one off, or incorrect? ").strip().lower()
        if raw_result in choices:
            return choices[raw_result]
        print("Please enter correct, one off, or incorrect.")


def main() -> None:
    bot = ConnectionBot(_read_words())
    solved_groups = 0

    while solved_groups < 4:
        guess = bot.get_guess()
        print("\nGuess:", ", ".join(guess))

        correctness = _read_correctness()
        bot.add_guess_result(guess, correctness)

        if correctness == ConnectionBot.Correctness.Correct:
            solved_groups += 1
            print(f"Marked correct. {4 - solved_groups} group(s) remaining.")

    print("All four groups marked correct.")


if __name__ == "__main__":
    main()
