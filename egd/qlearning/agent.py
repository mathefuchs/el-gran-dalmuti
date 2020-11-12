import numpy as np
import pandas as pd

from egd.game.state import has_finished, NUM_PLAYERS
from egd.game.moves import possible_next_moves
from egd.qlearning.qtable import QTable


class QLearningAgent:

    agents_finished = 0

    def __init__(self, playerIndex):
        """ Initialize an agent. """

        self._alpha = 0.5  # learning rate
        self._gamma = 0.95  # favour future rewards
        self._rewards = {
            0: 1.0,  # No other agent finished before
            1: 0.6,  # One other agent finished before
            2: 0.5,  # Two other agents finished before
            3: -1.0,  # Three other agents finished before
        }

        self._qtable = QTable()
        self._playerIndex = playerIndex
        self._debug = False

    def start_episode(self, initial_hand, num_episode=0):
        """ Initialize game with assigned initial hand. """

        self._hand = initial_hand
        self._num_episode = num_episode
        # amount of random decisions
        self._epsilon = 1 / np.sqrt(num_episode / 5 + 1)
        QLearningAgent.agents_finished = 0

    def save_model(self, to_path):
        """ Save the model to the specified path. """

        self._qtable.qtable.to_csv(to_path, header=None)

    def load_model(self, file_path):
        """ Load model from file. """

        self._qtable.restore_from_file(file_path)

    def do_step(self, already_played, board, always_use_best=False):
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
        if len(possible_hands) == 0:
            return False, already_played, board

        # Retrieve Q-Table for current state and add new if necessary
        learned_values = self._qtable.get_qtable_entry(
            already_played, board, self._hand)
        if not np.any(learned_values != None):
            self._qtable.create_qtable_entry(
                already_played, board, self._hand)
            learned_values = self._qtable.get_qtable_entry(
                already_played, board, self._hand)

        if self._debug:
            print("Player", self._playerIndex,
                  "- Learned Values", learned_values)

        # Do random decisions with a fixed probability
        if not always_use_best and np.random.uniform() < self._epsilon:
            action_index = np.random.choice(len(possible_hands))
        else:
            # Get best action with random tie-breaking
            possible_qvalues = learned_values.iloc[
                0, list(range(len(possible_hands)))
            ]
            action_index = np.random.choice(
                np.flatnonzero(np.isclose(
                    possible_qvalues, np.nanmax(possible_qvalues)
                )))

        # Compute next state
        next_hand = possible_hands[action_index]
        next_board = possible_boards[action_index]
        next_already_played = already_played + next_board \
            if not np.all(next_board == board) else already_played

        # Retrieve next state's q-value
        next_qvalues = self._qtable.get_qtable_entry(
            next_already_played, next_board, next_hand)
        next_max = np.nanmax(next_qvalues) \
            if np.any(next_qvalues != None) else 0

        # Determine reward
        if has_finished(next_hand):
            reward_earned = self._rewards[QLearningAgent.agents_finished]
            QLearningAgent.agents_finished += 1
        else:
            reward_earned = 0

        # Determine new value
        def update_func(old_qvalues):
            old_qvalue = old_qvalues.iloc[0, action_index]
            new_value = (1 - self._alpha) * old_qvalue + \
                self._alpha * (reward_earned + self._gamma * next_max)
            old_qvalues.iloc[0, action_index] = new_value
            return old_qvalues

        self._qtable.update_qtable(
            already_played, board, self._hand, update_func)

        # Return next state
        self._hand = next_hand
        return has_finished(self._hand), next_already_played, next_board
