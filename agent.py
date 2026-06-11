from game import Guess

class Agent:
    def __init__(self):
        # Agents must implement a constructor with either no arugments or all default arguments
        pass

    def setup_game(self, words: list[str]):
        # This method provides the agent with the 16 words of the connections game
        raise NotImplementedError()

    def get_guess(self) -> list[str] :
        # This method should return a list of 4 strings representing the words to be guessed
        raise NotImplementedError()
    
    def store_result(self, result: Guess) -> None :
        # This method does whatever the agent needs to do to process the result of the previous guess
        # Guess is an Enum:
        # CORRECT
        # ONE_AWAY
        # INCORRECT
        # ILL_FORMED
        raise NotImplementedError()

    def teardown_game(self):
        # This method is called when the agent is no longer required (game has ended)
        # and can be passed if agent is not memory-hungry
        raise NotImplementedError()