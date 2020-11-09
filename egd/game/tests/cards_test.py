import numpy as np
import unittest

from egd.game.cards import get_cards_array, NUM_CARD_VALUES


class CardsTest(unittest.TestCase):

    def test_card_array_zero_vector(self):
        """ Test with zero vector representations. """

        self.assertTrue(np.all(get_cards_array(
            4, 0) == np.zeros(NUM_CARD_VALUES)))
        self.assertTrue(np.all(get_cards_array(
            0, 0) == np.zeros(NUM_CARD_VALUES)))
        self.assertFalse(np.all(get_cards_array(
            0, 4) == np.zeros(NUM_CARD_VALUES)))

    def test_card_array_non_zero_vector(self):
        """ Test with non-zero vector representations. """

        self.assertTrue(np.all(get_cards_array(1, 2) == np.array(
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))
        self.assertTrue(np.all(get_cards_array(4, 3) == np.array(
            [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0])))
        self.assertFalse(np.all(get_cards_array(4, 3) == np.array(
            [0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0])))


if __name__ == '__main__':
    unittest.main()
