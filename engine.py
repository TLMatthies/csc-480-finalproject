from game import Game, Guess, BadGuess
from agent import Agent
import pandas as pd

#input_file = "data/Connections_Data.parquet"
input_file = "data/AI_Slop_15000.parquet"

class Engine:
    agent: Agent = Agent() # default agent will throw errors
    games: list[Game] = []

    def __init__(self, agent: Agent):
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
    
    def load_games(self, filename: str):
        # No sanitation, pandas will throw an error before anything bad happens
        game_data: pd.DataFrame = pd.read_parquet(filename)
        game_count = 0

        for _, data in game_data.groupby("Game ID"):
            words = data.sort_values("Group Level")["Word"].tolist()
            game_count = self.add_game(words)
        print("Loaded "+str(game_count)+" games.")
        return game_count
    
    def run_games(self, verbose=False):
        # Keep calling for more guesses on each game until game.is_solved()
        for game in self.games:
            if verbose:
                print("Starting Game "+str(game.game_id)+":")
            self.agent.setup_game(game.words)
            while (not game.is_solved()) and (game.guesses_remaining > 0):
                guess = self.agent.get_guess()
                result = self.submit_guess(game.game_id, guess)
                if verbose:
                    print("Guessing: "+str(guess)+" - "+str(result))
                self.agent.store_result(result)