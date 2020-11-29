import numpy as np
import pandas as pd
import unittest

from egd.game.cards import NUM_CARD_VALUES, get_cards_array
from egd.game.state import random_initial_cards
from egd.deep_qlearning.deepq_agent import DeepQAgent


class DeepQAgentTest(unittest.TestCase):

    def test_convert_to_batch(self):
        """ Tests the conversion to feed batch. """

        already_played = np.array([0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0])
        board = np.array([0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        hand = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        action = np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        agent = DeepQAgent(0)
        agent.start_episode(hand, 0)

        x = agent.convert_to_data_batch(already_played, board, hand, action)
        self.assertTrue(np.all(x == np.array([[
            [[0], [0], [2], [0], [4], [0], [0], [0], [0], [0], [0], [0], [0]],
            [[0], [0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
            [[0], [2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
        ]])))

    def test_init_fit_predict_model(self):
        """ Performs basic model operations. """

        already_played = np.array([0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0])
        board = np.array([0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        hand = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        action = np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        agent = DeepQAgent(0)
        agent.start_episode(hand, 0)

        # Fit observation to network
        agent.fit_value_to_network(
            already_played, board, hand, action,
            1.0, weight=100  # Fit 100 times to test
        )

        # Predict for same situation
        pred_q_value = agent.predict_q_value_from_network(
            already_played, board, hand, action
        )

        # Prediction should eventually be close to 1
        self.assertLess(pred_q_value, 1.0)
        self.assertGreater(pred_q_value, 0.5)


if __name__ == '__main__':
    unittest.main()
