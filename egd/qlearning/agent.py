import numpy as np
import pandas as pd

from egd.game.state import has_finished, NUM_PLAYERS
from egd.game.moves import possible_next_moves
from egd.qlearning.qtable import (
    create_qtable_df,
    create_qtable_entry,
    get_qtable_entry,
    update_qtable,
    delete_qtable_entry
)


class QLearningAgent:

    agents_finished = 0

    def __init__(self, playerIndex):
        """ Initialize an agent. """

        self._alpha = 0.8
        self._gamma = 1.0
        self._epsilon = 0.1
        self._rewards = {
            0: 1.0,  # No other agent finished before
            1: 0.6,  # One other agent finished before
            2: 0.4,  # Two other agents finished before
            3: 0.2,  # Three other agents finished before
        }

        self._qtable = create_qtable_df()
        self._playerIndex = playerIndex
        self._debug = True

    def start_episode(self, initial_hand, num_episode):
        """ Initialize game with assigned initial hand. """

        self._hand = initial_hand
        self._num_episode = num_episode
        QLearningAgent.agents_finished = 0

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
        learned_values = get_qtable_entry(
            self._qtable, already_played, board, self._hand)
        if not learned_values:
            self._qtable = create_qtable_entry(
                self._qtable, already_played, board, self._hand)
            learned_values = get_qtable_entry(
                self._qtable, already_played, board, self._hand)

        if self._debug:
            print("Player", self._playerIndex,
                  "- Learned Values", learned_values)

        # Do random decisions with a fixed probability
        if not always_use_best and np.random.uniform() < self._epsilon:
            action_index = np.random.choice(len(possible_hands))
        else:
            # Get best action with random tie-breaking
            possible_qvalues = learned_values.iloc[
                :, list(range(len(possible_hands)))
            ]
            action_index = np.random.choice(
                np.flatnonzero(np.isclose(
                    possible_qvalues, np.nanmax(possible_qvalues)
                )))

        # Compute next state
        next_hand = possible_hands[action_index]
        next_board = possible_boards[action_index]
        next_already_played = already_played + next_board

        # TODO -----------------------------------------------------------
        # Retrieve next state's q-value
        old_qvalue = learned_values.iloc[0, action_index]
        actions_possible_in_next_state = (next_state == 0)
        next_qvalues = player_table[np.all(
            player_states == next_state, axis=1)]
        next_max = np.nanmax(np.where(actions_possible_in_next_state, next_qvalues[0], np.nan)) \
            if next_qvalues.size > 0 else 0
        # TODO -----------------------------------------------------------

        # Determine reward
        if has_finished(next_hand):
            reward_earned = self._rewards[QLearningAgent.agents_finished]
            QLearningAgent.agents_finished += 1
        else:
            reward_earned = 0

        # TODO -----------------------------------------------------------
        # Determine new value
        new_value = (1 - self._alpha) * old_qvalue + \
            self._alpha * (reward_earned + self._gamma * next_max)
        learned_values[action] = new_value
        player_table[np.all(player_states == player_state,
                            axis=1)] = learned_values
        # TODO -----------------------------------------------------------

        # Return next state
        self._hand = next_hand
        return has_finished(self._hand), next_already_played, next_board
