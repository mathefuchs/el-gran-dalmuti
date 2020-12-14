import numpy as np
import pandas as pd

from egd.game.state import has_finished
from egd.game.moves import possible_next_moves


class RandomAgent:

    def __init__(self, playerIndex):
        """ Initialize an agent. """

        self.playerIndex = playerIndex

    def start_episode(self, initial_hand, num_episode=0):
        """ Initialize game with assigned initial hand. """

        self.hand = initial_hand

    def save_model(self):
        """ Save the model to the specified path. """

        # NOP
        pass

    def load_model(self):
        """ Load model from file. """

        # NOP
        pass

    def evaluate_inference_mode(self):
        """ No special evaluation needed. """

        pass

    def do_step(self, already_played, board, agents_finished,
                next_action_wins_board=lambda a, b: False,
                always_use_best=True, print_luck=False):
        """
            Performs a step in the game.

            Returns (Player finished, Already played cards, New board)
        """

        # If player has already finished, pass
        if has_finished(self.hand):
            return True, already_played, board, False

        # Possible actions; Pass if no possible play
        possible_actions = possible_next_moves(self.hand, board)
        if len(possible_actions) == 1 and \
                np.all(possible_actions[0] == 0):
            return False, already_played, board, False

        # Decide randomly
        action_index = np.random.randint(len(possible_actions))
        action_taken = possible_actions[action_index]

        # Compute next state
        next_hand = self.hand - action_taken
        next_board = board if np.all(action_taken == 0) else action_taken
        next_already_played = already_played + action_taken

        # Return next state
        self.hand = next_hand
        return (has_finished(self.hand), next_already_played,
                next_board, True)
