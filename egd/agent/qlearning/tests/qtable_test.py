import numpy as np
import pandas as pd
import unittest

from egd.game.cards import NUM_CARD_VALUES
from egd.agent.qlearning.qtable import QTable


class QTableTest(unittest.TestCase):

    def setUp(self):
        """ Setup QTable. """

        self.qtable = QTable()

        self.already_played = np.random.randint(
            NUM_CARD_VALUES, size=QTable.single_state_width)
        self.board = np.random.randint(
            NUM_CARD_VALUES, size=QTable.single_state_width)
        self.hand = np.random.randint(
            NUM_CARD_VALUES, size=QTable.single_state_width)

    def test_insert_retrieve(self):
        """ Insert single row and retrieve it. """

        # Check insertion
        self.assertFalse(self.qtable.get_qtable_entry(
            self.already_played, self.board, self.hand))
        self.qtable.create_qtable_entry(
            self.already_played, self.board, self.hand)
        self.assertTrue(np.all(self.qtable.get_qtable_entry(
            self.already_played, self.board, self.hand) == 0))

        # Check order of args
        self.assertFalse(self.qtable.get_qtable_entry(
            self.already_played, self.already_played, self.hand))
        self.assertFalse(self.qtable.get_qtable_entry(
            self.board, self.already_played, self.hand))
        self.assertFalse(self.qtable.get_qtable_entry(
            self.hand, self.hand, self.hand))

    def test_insert_update(self):
        """ Insert and update row. """

        # Check insertion and value
        self.qtable.create_qtable_entry(
            self.already_played, self.board, self.hand)
        self.assertTrue(np.all(self.qtable.get_qtable_entry(
            self.already_played, self.board, self.hand) == 0))

        # Update and check again
        def update_func(array):
            array.iloc[0, 0] = 11.0
            return array + 1

        self.qtable.update_qtable(
            self.already_played, self.board,
            self.hand, update_func)
        row = self.qtable.get_qtable_entry(
            self.already_played, self.board, self.hand)
        self.assertEqual(row.iloc[0, 0], 12)
        self.assertEqual(row.iloc[0, 4], 1)

    def test_insert_delete(self):
        """ Insert and delete row. """

        # Check insertion and value
        self.qtable.create_qtable_entry(
            self.already_played, self.board, self.hand)
        self.assertTrue(np.all(self.qtable.get_qtable_entry(
            self.already_played, self.board, self.hand) == 0))

        # Delete row
        self.qtable.delete_qtable_entry(
            self.already_played, self.board, self.hand)
        self.assertFalse(self.qtable.get_qtable_entry(
            self.already_played, self.board, self.hand))


if __name__ == '__main__':
    unittest.main()
