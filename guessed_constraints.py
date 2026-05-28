from z3 import Solver, Sum, Int, Or, Not, And, If, sat as z3sat, ArithRef
from enum import Enum

class PrevConstraint():
    class Correctness(Enum):
        FullyWrong = 0
        OneOff = 1
        Correct = 2

    def __init__(self, words: list[str]) -> None:
        if len(words) != 16 or len(set(words)) != 16:
            raise ValueError("Array needs a length of 16 with unique values!")

        self.words: dict[str, ArithRef] = {}
        self.solver = Solver()

        for word in words:
            self.words[word] = Int(word)
            self.solver.add(self.words[word] >= 1, self.words[word] <= 4)

        self.solver.add(Sum([If(word == 1, 1, 0) for word in self.words.values()])==4)
        self.solver.add(Sum([If(word == 2, 1, 0) for word in self.words.values()])==4)
        self.solver.add(Sum([If(word == 3, 1, 0) for word in self.words.values()])==4)
        self.solver.add(Sum([If(word == 4, 1, 0) for word in self.words.values()])==4)
    
    def is_possible(self, word1: str, word2: str, word3: str, word4: str) -> bool:
        """
        Check if a guess is possible, given what previous guesses we've taken and their results.
        Example: previously guessed a, b, c, d and told completely wrong, this will return false 
        if a, b, c, e is checked.
        """
        if len(set([word1, word2, word3, word4])) != 4:
            raise ValueError("PrevConstraint.is_possible() needs 4 unique words")

        try:
            self.solver.push()
            self.solver.add(self.words[word1] == self.words[word2],
                            self.words[word2] == self.words[word3],
                            self.words[word3] == self.words[word4])
        except:
            self.solver.pop()
            raise ValueError("Guessed words need to be in the original set!")
        result = (self.solver.check() == z3sat)
        self.solver.pop()
        return result

    def add_guess(self, word1: str, word2: str, word3: str, word4: str, correctness: Correctness) -> None:
        if correctness == self.Correctness.Correct:
            self.solver.add(And(self.words[word1] == self.words[word2],
                                self.words[word2] == self.words[word3],
                                self.words[word3] == self.words[word4]))
                            
        elif correctness == self.Correctness.OneOff:
            self.solver.add(Or(
                And(self.words[word1] == self.words[word2], 
                    self.words[word2] == self.words[word3],
                    self.words[word3] != self.words[word4]),
                And(self.words[word1] == self.words[word2],
                    self.words[word2] == self.words[word4],
                    self.words[word4] != self.words[word3]),
                And(self.words[word1] == self.words[word3],
                    self.words[word3] == self.words[word4],
                    self.words[word4] != self.words[word2]),
                And(self.words[word2] == self.words[word3],
                    self.words[word3] == self.words[word4],
                    self.words[word4] != self.words[word1])))
        elif correctness == self.Correctness.FullyWrong:
            self.solver.add(Not(Or(
                And(self.words[word1] == self.words[word2], 
                    self.words[word2] == self.words[word3],
                    self.words[word3] != self.words[word4]),
                And(self.words[word1] == self.words[word2],
                    self.words[word2] == self.words[word4],
                    self.words[word4] != self.words[word3]),
                And(self.words[word1] == self.words[word3],
                    self.words[word3] == self.words[word4],
                    self.words[word4] != self.words[word2]),
                And(self.words[word2] == self.words[word3],
                    self.words[word3] == self.words[word4],
                    self.words[word4] != self.words[word1]))))
            self.solver.add(Not(And(
                self.words[word1] == self.words[word2], 
                self.words[word2] == self.words[word3],
                self.words[word3] == self.words[word4]
            )))
        else:
            raise ValueError("Not a recognized value of correctness.")
            


test1 = PrevConstraint(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'])
test1.add_guess('a', 'b', 'c', 'd', PrevConstraint.Correctness.OneOff)
assert test1.is_possible('a', 'b', 'c', 'd') == False
assert test1.is_possible('a', 'b', 'c', 'e') == True
assert test1.is_possible('a', 'b', 'e', 'f') == False
assert test1.is_possible('e', 'a', 'b', 'c') == True
assert test1.is_possible('a', 'e', 'c', 'd') == True

test2 = PrevConstraint(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'])
test2.add_guess('a', 'b', 'c', 'd', PrevConstraint.Correctness.FullyWrong)
assert test2.is_possible('a', 'b', 'c', 'd') == False
assert test2.is_possible('a', 'b', 'c', 'e') == False
assert test2.is_possible('a', 'b', 'e', 'f') == True

test3 = PrevConstraint(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'])
test3.add_guess('a', 'b', 'c', 'd', PrevConstraint.Correctness.Correct)
assert test3.is_possible('a', 'b', 'c', 'e') == False
assert test3.is_possible('a', 'f', 'g', 'h') == False
assert test3.is_possible('e', 'f', 'g', 'h') == True


