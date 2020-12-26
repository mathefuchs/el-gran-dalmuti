import numpy as np
import pandas as pd

from egd.agent.base_agent import ModelBase
from egd.game.cards import get_cards_array, JOKER
from egd.game.state import has_finished, NUM_PLAYERS
from egd.game.moves import possible_next_moves


class HumanAgent(ModelBase):

    def prepare_step(self):
        """ Prepares the step to do. """

        # Show prompt for action
        print("It's your turn Player", self.playerIndex)
        print("                   1 2 3 4 5 6 7 8 9 . . . J")
        print("Your hand:       ", self.hand)

    def get_action_values(self, possible_actions: np.ndarray) -> np.ndarray:
        """ Retrieves the values for all provided actions.

        Args:
            possible_actions (np.ndarray): Possible actions

        Returns:
            np.ndarray: Values for all provided actions
        """

        # TODO Refactor method
        # Ask for action
        while True:
            cmd = input(
                "Enter your move (<card_type> <num_cards> [<num_jokers>]; or 'pass'): ")

            if cmd == "pass":
                if not np.all(self.state.curr_board == 0):
                    action_index = 0
                    break
                else:
                    print("Invalid move.")
                    continue

            try:
                cards_to_play = list(map(int, cmd.split()))
            except:
                print("Invalid move.")
                continue

            if len(cards_to_play) == 2 and cards_to_play[0] >= 1 \
                    and cards_to_play[0] <= 12 and cards_to_play[1] >= 1 \
                    and cards_to_play[1] <= self.hand[cards_to_play[0] - 1]:
                card_array_to_play = get_cards_array(
                    cards_to_play[0] - 1, cards_to_play[1])
            elif len(cards_to_play) == 3 and cards_to_play[0] >= 1 \
                    and cards_to_play[0] <= 12 and cards_to_play[1] >= 0 \
                    and cards_to_play[1] <= self.hand[cards_to_play[0] - 1] \
                    and cards_to_play[2] in [0, 1, 2] \
                    and cards_to_play[1] + cards_to_play[2] >= 1:
                card_array_to_play = get_cards_array(
                    cards_to_play[0] - 1, cards_to_play[1]) + \
                    get_cards_array(JOKER, cards_to_play[2])
            else:
                print("Invalid move.")
                continue

            if np.any(np.all(card_array_to_play == possible_actions, axis=1)) \
                    and not np.all(card_array_to_play == self.state.curr_board):
                action_index = np.where(np.all(
                    card_array_to_play == possible_actions, axis=1))[0][0]
                break
            else:
                print("Invalid move.")
                continue

        # Return unit vector with choice
        action_values = np.zeros(possible_actions.shape[0], dtype=np.int8)
        action_values[action_index] = 1
        return action_values
