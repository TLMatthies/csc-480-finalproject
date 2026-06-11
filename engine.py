from game import Game, Guess, BadGuess
from agent import Agent
import pandas as pd
from collections.abc import Callable

#input_file = "data/Connections_Data.parquet"
input_file = "data/AI_Slop_15000.parquet"

class Engine:

    class GameState:
        game: Game
        agent: Agent
        solved: bool
        degree: int = -1 # either guesses remaining (if solved) or categories solved (if not solved)

        def __init__(self, game: Game, agent: Agent):
            self.game = game
            self.agent = agent

    agentFactory: Callable[[],Agent] # Can pass basic factory using "lambda: NamedAgent(data)"
    games: list[GameState]
    guesses_per_game = 4

    def __init__(self, factory: Callable[[], Agent], guesses:int = 4):
        self.agentFactory = factory
        self.guesses_per_game = guesses
        self.games = []
    
    def add_game(self, words: list[str], guesses:int = 4):
        game_id = len(self.games)
        self.games.append(self.GameState(Game(game_id, words, guesses),self.agentFactory()))
        return game_id
    
    def submit_guess(self, game_id:int, words: list[str]):
        try:
            return self.games[game_id].game.addGuess(words)
        except BadGuess as bg:
            print("Guess "+str(words)+" could not be submitted:\n"+str(bg))
            return Guess.ILL_FORMED
    
    def is_game_solved(self, game_id: int):
        return self.games[game_id].game.is_solved()
    
    def load_games(self, filename: str):
        # No sanitation, pandas will throw an error before anything bad happens
        game_data: pd.DataFrame = pd.read_parquet(filename)
        game_count = 0
        for _, data in game_data.groupby("Game ID"):
            words = data.sort_values("Group Level")["Word"].tolist()
            game_count = self.add_game(words)
            if game_count%100 == 0:
                print("\rLoading "+str(game_count)+" games.",end='', flush=True)
        print("\rLoaded "+str(game_count+1)+" games.   ")
        return game_count+1
    
    def run_games(self, verbose=False):
        solved = [0]*self.guesses_per_game
        unsolved = [0,0,0,0]
        for game in self.games:
            if not verbose and game.game.game_id%100 == 0:
                print("\rGame: "+str(game.game.game_id), end='',flush=True)
            self.run_game(game.game.game_id, verbose)
            if game.solved:
                solved[game.degree-1] += 1
            else:
                unsolved[game.degree] += 1
        total_solved = sum(solved)
        total_unsolved = sum(unsolved)
        print("\rGames finished!")
        print(str(total_solved) + " games solved: " + str(solved))
        print(str(total_unsolved) + " games failed: "+ str(unsolved))
        
    
    def run_game(self, game_id:int, verbose=False):
        game = self.games[game_id].game
        agent = self.games[game_id].agent
        if verbose:
            print("Game "+str(game.game_id)+" starting...")
        agent.setup_game(game.words)
        # Keep calling get_guess until it runs out of guesses or solves it!
        # Note: correct answers do not cost a guess (refunded by game)
        while (not game.is_solved()) and (game.guesses_remaining > 0):
            guess = agent.get_guess()
            result = self.submit_guess(game_id, guess)
            if verbose:
                print("             Guessing: "+str(guess)+" - "+str(result))
            agent.store_result(result)
        solved_categories = 0
        for category in game.answers:
            if category.isSolved:
                solved_categories += 1
        if solved_categories == 4:
            self.games[game_id].solved = True
            self.games[game_id].degree = game.guesses_remaining
        else:
            self.games[game_id].solved = False
            self.games[game_id].degree = solved_categories