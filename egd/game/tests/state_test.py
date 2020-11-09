import numpy as np
import unittest

from egd.game.cards import NUM_CARD_VALUES, AVAILABLE_CARDS
from egd.game.state import has_already_won, random_initial_cards, NUM_PLAYERS


class StateTest(unittest.TestCase):

    def test_has_won(self):
        """ Tests whether a hand has won. """

        self.assertTrue(has_already_won(np.zeros(NUM_CARD_VALUES)))
        self.assertFalse(has_already_won(np.ones(NUM_CARD_VALUES)))
        self.assertTrue(np.all(has_already_won(
            np.zeros((2, NUM_CARD_VALUES))) == np.array([True, True])))
        self.assertTrue(np.all(
            has_already_won(np.array([
                [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]
            ])) == np.array([
                False, True, False
            ])))

    def test_initial_state(self):
        """ Tests the initial distribution of cards. """

        # Check that randomness only depends upon numpy
        np.random.seed(42)
        self.assertTrue(np.all(random_initial_cards()
                               == random_initial_cards()))

        # Check number of cards of each kind
        self.assertTrue(
            np.all(np.sum(random_initial_cards(), axis=0) == AVAILABLE_CARDS))

        # Check number of cards for each player
        cards_per_player = np.sum(AVAILABLE_CARDS) // NUM_PLAYERS
        self.assertTrue(np.all(np.sum(random_initial_cards(), axis=1)
                               == cards_per_player * np.ones(NUM_PLAYERS)))


if __name__ == '__main__':
    unittest.main()
