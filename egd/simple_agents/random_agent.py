import numpy as np
import pandas as pd

from egd.game.state import has_finished
from egd.game.moves import possible_next_moves


class RandomAgent:

    def __init__(self, playerIndex):
        """ Initialize an agent. """

        self._playerIndex = playerIndex

    def start_episode(self, initial_hand, num_episode=0):
        """ Initialize game with assigned initial hand. """

        self._hand = initial_hand

    def save_model(self):
        """ Save the model to the specified path. """

        # NOP
        pass

    def load_model(self):
        """ Load model from file. """

        # NOP
        pass

    def do_step(self, already_played, board, agents_finished,
                always_use_best=True, print_luck=False):
        """
            Performs a step in the game.

            Returns (Player finished, Already played cards, New board)
        """

        # If player has already finished, pass
        if has_finished(self._hand):
            return True, already_played, board

        # Possible actions; Pass if no possible play
        possible_hands, possible_boards = \
            possible_next_moves(self._hand, board)
        if len(possible_hands) == 1 and \
                np.all(possible_boards[0] == board):
            return False, already_played, board

        # Decide randomly
        action_index = np.random.randint(len(possible_hands))

        # Compute next state
        next_hand = possible_hands[action_index]
        next_board = possible_boards[action_index]
        next_already_played = already_played + next_board \
            if not np.all(next_board == board) else already_played

        # Return next state
        self._hand = next_hand
        return has_finished(self._hand), next_already_played, next_board
