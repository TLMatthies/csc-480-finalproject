import zGuesser
from game import Game, Guess, BadGuess

class Engine:
    agent = None
    games: list[Game] = []

    def __init__(self, agent):
        self.agent = agent
    
    def add_game(self, words: list[str], guesses:int = 4):
        game_id = len(self.games)
        self.games.append(Game(game_id, words, guesses))
        return game_id
    
    def submit_guess(self, game_id:int, words: list[str]):
        try:
            return self.games[game_id].addGuess(words)
        except BadGuess as bg:
            print("Guess "+str(words)+" could not be submitted:\n"+str(bg))
            return Guess.ILL_FORMED
    
    def is_game_solved(self, game_id: int):
        return self.games[game_id].is_solved()
        