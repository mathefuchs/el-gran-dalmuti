import numpy as np
import pandas as pd

from egd.game.cards import get_cards_array, JOKER
from egd.game.state import has_finished, NUM_PLAYERS
from egd.game.moves import possible_next_moves


class HumanAgent:

    def __init__(self, playerIndex):
        """ Initialize an agent. """

        self._playerIndex = playerIndex

    def start_episode(self, initial_hand):
        """ Initialize game with assigned initial hand. """

        self._hand = initial_hand

    def do_step(self, already_played, board, always_use_best=True, print_luck=False):
        """
            Performs a step in the game based on human input.

            Returns (Player finished, Already played cards, New board)
        """

        # Show prompt for action
        print("It's your turn Player", self._playerIndex)
        print("                   1 2 3 4 5 6 7 8 9 . . . J")
        print("Your hand:       ", self._hand)

        # If player has already finished, pass
        if has_finished(self._hand):
            return True, already_played, board

        # Possible actions; Pass if no possible play
        possible_hands, possible_boards = \
            possible_next_moves(self._hand, board)
        if len(possible_hands) == 0:
            return False, already_played, board

        # Ask for action
        while True:
            cmd = input(
                "Enter your move (<card_type> <num_cards> [<num_jokers>]; or 'pass'): ")

            if cmd == "pass":
                if not np.all(board == 0):
                    return False, already_played, board
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
                    and cards_to_play[1] <= self._hand[cards_to_play[0] - 1]:
                card_array_to_play = get_cards_array(
                    cards_to_play[0] - 1, cards_to_play[1])
            elif len(cards_to_play) == 3 and cards_to_play[0] >= 1 \
                    and cards_to_play[0] <= 12 and cards_to_play[1] >= 0 \
                    and cards_to_play[1] <= self._hand[cards_to_play[0] - 1] \
                    and cards_to_play[2] in [0, 1, 2] \
                    and cards_to_play[1] + cards_to_play[2] >= 1:
                card_array_to_play = get_cards_array(
                    cards_to_play[0] - 1, cards_to_play[1]) + \
                    get_cards_array(JOKER, cards_to_play[2])
            else:
                print("Invalid move.")
                continue

            if np.any(np.all(card_array_to_play == possible_boards, axis=1)) \
                    and not np.all(card_array_to_play == board):
                self._hand -= card_array_to_play
                next_board = card_array_to_play
                next_already_played = already_played + next_board
                return has_finished(self._hand), next_already_played, next_board
            else:
                print("Invalid move.")
                continue
