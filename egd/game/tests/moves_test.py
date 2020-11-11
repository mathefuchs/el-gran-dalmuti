import numpy as np
import pandas as pd
import unittest

from egd.game.cards import NUM_CARD_VALUES, AVAILABLE_CARDS
from egd.game.moves import possible_next_moves


class MovesTest(unittest.TestCase):

    def test_possible_next_moves_empty_board(self):
        """ Tests possible moves on empty board. """

        hand, board = possible_next_moves(
            np.array([0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2]),
            np.zeros(NUM_CARD_VALUES))
        self.assertTrue(np.all(hand == np.array([
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 1],
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2],
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 2],
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 1],
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 2],
            [0, 2, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1],
            [0, 2, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2],
            [0, 1, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0]
        ])))
        self.assertTrue(np.all(board == np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
        ])))

    def test_possible_next_moves_empty_board_lots_of_cards(self):
        """ Tests possible moves on empty board with lots of cards. """

        hand, board = possible_next_moves(
            AVAILABLE_CARDS,
            np.zeros(NUM_CARD_VALUES))

        true_hand = pd.read_csv(
            "./egd/game/tests/test-data/hand.csv", header=None).values
        true_board = pd.read_csv(
            "./egd/game/tests/test-data/board.csv", header=None).values

        self.assertTrue(np.all(hand == true_hand))
        self.assertTrue(np.all(board == true_board))

    def test_possible_next_moves_non_empty_board(self):
        """ Tests possible next moves. """

        hand, board = possible_next_moves(
            np.array([0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]))
        self.assertTrue(np.all(hand == np.array([
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2],
            [0, 2, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2],
            [0, 1, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 1]
        ])))
        self.assertTrue(np.all(board == np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])))

        hand, board = possible_next_moves(
            np.array([1, 2, 3, 1, 0, 0, 0, 3, 0, 4, 0, 0, 2]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1]))
        self.assertTrue(np.all(hand == np.array([
            [1, 2, 3, 1, 0, 0, 0, 3, 0, 4, 0, 0, 2],
            [1, 2, 3, 1, 0, 0, 0, 0, 0, 4, 0, 0, 2],
            [1, 2, 3, 1, 0, 0, 0, 1, 0, 4, 0, 0, 1],
            [1, 2, 3, 1, 0, 0, 0, 2, 0, 4, 0, 0, 0],
            [1, 2, 3, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0],
            [1, 2, 0, 1, 0, 0, 0, 3, 0, 4, 0, 0, 2],
            [1, 2, 1, 1, 0, 0, 0, 3, 0, 4, 0, 0, 1],
            [1, 2, 2, 1, 0, 0, 0, 3, 0, 4, 0, 0, 0],
            [1, 0, 3, 1, 0, 0, 0, 3, 0, 4, 0, 0, 1],
            [1, 1, 3, 1, 0, 0, 0, 3, 0, 4, 0, 0, 0],
            [0, 2, 3, 1, 0, 0, 0, 3, 0, 4, 0, 0, 0]
        ])))
        self.assertTrue(np.all(board == np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
        ])))


if __name__ == '__main__':
    unittest.main()
