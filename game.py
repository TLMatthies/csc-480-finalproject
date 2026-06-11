import random, copy
from enum import Enum

class Guess(Enum):
    CORRECT = 0
    ONE_AWAY = 1
    INCORRECT = 2
    ILL_FORMED = -1

class BadGuess(ValueError):
    pass

class Game:

    class Category:
        words:list[str] = []
        word_set: frozenset[str] = frozenset()
        difficulty: int = -1
        guesses_remaining: int = 4
        isSolved: bool = False

        def __init__(self, diff: int, words: list[str]):
            if len(words) != 4:
                raise ValueError("Categories must contain exactly 4 words:\n"+str(words))
            words = [word.lower() for word in words]
            self.words = words
            self.word_set = frozenset(words)
            if len(self.words) != len(self.word_set):
                raise ValueError("Categories must contain only unique words:\n"+str(words))
            self.difficulty = diff
        
        def solve(self):
            self.isSolved = True

    words: list[str] = [] # all 16 words
    answers: list[Category] = [] # 4 Categorys
    guesses_remaining: int = 4
    previous_guesses: dict[frozenset[str], Guess] = {}

    def __init__(self, ID: int, words: list[str], guesses: int = 4):
        self.game_id = ID
        if len(words) != 16:
            raise ValueError("Games must contain exactly 16 words.")
        self.words = words
        self.answers = []
        for w in range(0, 16, 4):
            categoryWords = words[w:w+4]
            category = self.Category(w, categoryWords)
            self.answers.append(category)
        
        self.previous_guesses = {}
        combined = len(set().union(*(answer.word_set for answer in self.answers)))
        if combined != 16:
            raise ValueError("All 16 words in a game must be unique.")
        if guesses < 1:
            raise ValueError("Don't be mean. You need at least 1 guess.")
        self.guesses_remaining = guesses
    
    def get_words(self):
        words = copy.deepcopy(self.words)
        random.shuffle(words)
        return words
    
    def get_words_remaining(self):
        words = [word for group in self.answers for word in group.words if not group.isSolved]
        random.shuffle(words)
        return words
    
    def is_solved(self):
        category_solved = True
        for category in self.answers:
            category_solved &= category.isSolved
        return category_solved

    def addGuess(self, words: list[str]):
        # Returns 1 if found category, 0 if 1-away, and -1 otherwise
        guessQuality = Guess.INCORRECT
        if len(words) != 4:
            raise BadGuess("Guesses must be exactly 4 words")
        for word in words:
            if not word in self.words:
                raise BadGuess("Guesses cannont include unknown word '"+str(word)+"'")
        word_set = frozenset(words)
        # Check the logical validity of the guess only after sanitiziation
        for guess, quality in self.previous_guesses.items():
            if word_set == guess:
                raise BadGuess("Guesses cannot be duplicates.")
            if len(word_set & guess) == 3 and quality != Guess.ONE_AWAY:
                raise BadGuess("Guesses must not be similar to known bad guesses.")
        for group in self.answers:
            if word_set == group.word_set:
                group.solve()
                guessQuality = Guess.CORRECT
                break
            elif len(word_set & group.word_set) == 3:
                guessQuality = Guess.ONE_AWAY
                break
        # Gotta record the new 'previous guess' before returning
        self.previous_guesses[word_set] = guessQuality
        if guessQuality == Guess.CORRECT:
            self.guesses_remaining += 1
        self.guesses_remaining -= 1
        return guessQuality