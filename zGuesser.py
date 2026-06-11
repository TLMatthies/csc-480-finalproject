from agent import Agent
from game import Guess
import pandas as pd
from itertools import combinations as comb
from z3 import Optimize, Int, If, Sum, Or, sat, ArithRef

class zGuesserAgent(Agent):

    class zGuesser:
        zWlookup: dict[tuple[str, str], float] = {}
        words: list[str] = []
        iWords: dict[str,ArithRef] = {}
        score_calc: ArithRef
        opt: Optimize = Optimize()
        comb_values: list[tuple[str,str,float]] = []
        solved_groups: list[frozenset[str]] = []
        last_guess: frozenset[str] = frozenset()

        def __init__(self, words: list[str], lookup: dict[tuple[str,str], float]):
            self.zWlookup = lookup
            self.words = words
            self.opt = Optimize()
            if len(self.words) != 16:
                raise ValueError("Agent cannot handle lists of size other than 16.")

            self.solved_groups = []
            self.last_guess = frozenset()
            # This probably isn't necessary, but done anyway.
            self.comb_values = []
            for w1, w2 in comb(self.words, 2):
                a,b = sorted([w1, w2])
                val = self.zWlookup.get((a,b), 0)
                self.comb_values.append((a,b,val))

            # Setting up basic sanitation rules
            self.iWords = {w: Int(f"group_{w}") for w in self.words}
            # Groups are 0,1,2,3
            for w in self.words:
                self.opt.add(self.iWords[w] >= 0)
                self.opt.add(self.iWords[w] < 4)
            # Only 4 words in a group
            for i in range(4):
                self.opt.add(Sum([If(self.iWords[w] == i, 1, 0) for w in self.words]) == 4)
            # Build score expression here, since it doesn't change between guesses:
            scores = []
            for a, b, val in self.comb_values:
                scores.append(If(self.iWords[a] == self.iWords[b], val, 0))
            self.score_calc = Sum(scores) #type:ignore
            # This ends all the rules which are independant of guesses

        def groupScore(self, words: list[str]) -> float:
            total = 0
            for w1, w2 in comb(words,2):
                a,b = sorted([w1,w2])
                total += self.zWlookup.get((a,b),0) #type:ignore
            return total
        
        def add_result(self, result: Guess):
            # Assumes that self.last_guess stores the relevant guess
            match result:
                case Guess.CORRECT:
                    self.solved_groups.append(self.last_guess)
                    for w in self.last_guess:
                        self.opt.add(self.iWords[w] == 3 - (len(self.solved_groups) - 1))
                case Guess.ONE_AWAY:
                    self.opt.add(Or([Sum([If(self.iWords[w] == g, 1, 0)
                                          for w in self.last_guess]) == 3
                                     for g in range(4)]))
                case Guess.INCORRECT:
                    for g in range(4):
                        self.opt.add(Sum([If(self.iWords[w] == g, 1, 0)
                                          for w in self.last_guess]) < 3)
                case _: # this also handles ill-formed guesses
                    raise ValueError("I made a bad guess? I'm a bad agent.")

        def makeGuess(self) -> list[str]: 
            self.opt.push() 
            self.opt.maximize(self.score_calc)

            if self.opt.check() == sat:
                model = self.opt.model()
                groups: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
                for w in self.words:
                    g: int = model[self.iWords[w]].as_long() #type:ignore
                    groups[g].append(w)
                groupScores: list[float] = []
                totalScore: float = 0
                for g, grouped in groups.items():
                    score = self.groupScore(grouped)
                    groupScores.append(score)
                    totalScore += score

                # print("I think you should guess:\n")
                # for g, grouped in groups.items():
                #     score = groupScores[g]
                #     percent = (100*score/totalScore if totalScore > 0 else 0)
                    # print(
                    #     f"{g} - "
                    #     f"{score}"
                    #     f"({percent}%): "
                    #     f"{', '.join(grouped)}"
                    # )
                
                self.opt.pop()
                self.last_guess = frozenset(groups[0])
                return groups[0]
            else:
                print("Oh dear, I can't figure out what I should guess!")
                self.opt.pop()
                return []

    zWlookup: dict[tuple[str,str],float] = {}
    agent: zGuesser | None = None

    def __init__(self, file_name: str):
        cls = type(self)
        if not cls.zWlookup:
            zWeights = pd.read_parquet(file_name)
            cls.zWlookup = {(row.Word1, row.Word2): row.Weight #type:ignore
                        for row in zWeights.itertuples(index=False)}

    def setup_game(self, words: list[str]):
        self.agent = self.zGuesser(words, self.zWlookup)

    def get_guess(self) -> list[str]:
        if not self.agent:
            raise ValueError("Cannot get guess before setting up the game.")
        return self.agent.makeGuess()
    
    def store_result(self, result: Guess) -> None :
        if not self.agent:
            raise ValueError("Cannot store guess before setting up the game.")
        self.agent.add_result(result)
    
    def teardown_game(self):
        self.agent = None