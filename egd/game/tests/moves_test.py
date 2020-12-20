import numpy as np
import pandas as pd
import unittest

from egd.game.cards import NUM_CARD_VALUES, AVAILABLE_CARDS
from egd.game.moves import possible_next_moves, only_passing_possible


class MovesTest(unittest.TestCase):

    def test_only_passing_true(self):
        """ Tests that only passing allowed. """

        self.assertTrue(only_passing_possible(
            np.zeros(NUM_CARD_VALUES), np.zeros(NUM_CARD_VALUES)
        ))
        self.assertTrue(only_passing_possible(
            np.array([0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0]),
            np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ))

    def test_possible_next_moves_empty_board(self):
        """ Tests possible moves on empty board. """

        hand = np.array([0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2])
        board = np.zeros(NUM_CARD_VALUES)
        actions = possible_next_moves(hand, board)
        only_passing = only_passing_possible(hand, board)

        self.assertFalse(only_passing)
        self.assertTrue(np.all(actions == np.array([
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

        hand = AVAILABLE_CARDS
        board = np.zeros(NUM_CARD_VALUES)
        actions = possible_next_moves(hand, board)
        only_passing = only_passing_possible(hand, board)

        true_actions = pd.read_csv(
            "./egd/game/tests/test-data/actions.csv", header=None).values

        self.assertFalse(only_passing)
        self.assertTrue(np.all(actions == true_actions))

    def test_possible_next_moves_non_empty_board(self):
        """ Tests possible next moves. """

        hand = np.array([0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2])
        board = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0])
        actions = possible_next_moves(hand, board)
        only_passing = only_passing_possible(hand, board)

        self.assertFalse(only_passing)
        self.assertTrue(np.all(actions == np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])))

        hand = np.array([1, 2, 3, 1, 0, 0, 0, 3, 0, 4, 0, 0, 2])
        board = np.array([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1])
        actions = possible_next_moves(hand, board)
        only_passing = only_passing_possible(hand, board)

        self.assertFalse(only_passing)
        self.assertTrue(np.all(actions == np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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

    def test_possible_next_moves_bug_jokers_only(self):
        """ Tests that a historic bug does not occur again. """

        hand = np.array([0, 1, 0, 0, 0, 1, 0, 0, 2, 4, 2, 4, 1])
        board = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        actions = possible_next_moves(hand, board)
        only_passing = only_passing_possible(hand, board)

        self.assertFalse(only_passing)
        self.assertTrue(np.all(actions == np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])))

    def test_possible_next_moves_bug_one_joker(self):
        """ Tests that a historic bug does not occur again. """

        hand = np.array([0, 1, 0, 0, 0, 1, 0, 0, 2, 4, 2, 4, 1])
        board = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        actions = possible_next_moves(hand, board)
        only_passing = only_passing_possible(hand, board)

        self.assertFalse(only_passing)
        self.assertTrue(np.all(actions == np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])))


if __name__ == '__main__':
    unittest.main()
