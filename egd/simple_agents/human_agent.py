import numpy as np
import pandas as pd

from egd.game.cards import get_cards_array, JOKER
from egd.game.state import has_finished, NUM_PLAYERS
from egd.game.moves import possible_next_moves


class HumanAgent:

    def __init__(self, playerIndex):
        """ Initialize an agent. """

        self.playerIndex = playerIndex

    def start_episode(self, initial_hand, num_epoch=0):
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
            Performs a step in the game based on human input.

            Returns (Player finished, Already played cards, New board)
        """

        # Show prompt for action
        print("It's your turn Player", self.playerIndex)
        print("                   1 2 3 4 5 6 7 8 9 . . . J")
        print("Your hand:       ", self.hand)

        # If player has already finished, pass
        if has_finished(self.hand):
            return True, already_played, board, False

        # Possible actions; Pass if no possible play
        possible_actions = possible_next_moves(self.hand, board)
        if len(possible_actions) == 1 and \
                np.all(possible_actions[0] == 0):
            return False, already_played, board, False

        # Ask for action
        while True:
            cmd = input(
                "Enter your move (<card_type> <num_cards> [<num_jokers>]; or 'pass'): ")

            if cmd == "pass":
                if not np.all(board == 0):
                    return False, already_played, board, False
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
                    and not np.all(card_array_to_play == board):
                self.hand -= card_array_to_play
                next_board = card_array_to_play
                next_already_played = already_played + next_board
                return has_finished(self.hand), next_already_played, next_board, False
            else:
                print("Invalid move.")
                continue
