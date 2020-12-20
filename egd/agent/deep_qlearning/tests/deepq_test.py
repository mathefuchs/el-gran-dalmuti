import numpy as np
import pandas as pd
import unittest

from egd.game.cards import NUM_CARD_VALUES, get_cards_array
from egd.game.state import random_initial_cards
from egd.agent.deep_qlearning.deepq_agent import DeepQAgent


class DeepQAgentTest(unittest.TestCase):

    def test_convert_to_batch(self):
        """ Tests the conversion to feed batch. """

        already_played = np.array([0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0])
        board = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        hand = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        actions = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        agent = DeepQAgent(0)
        agent.start_episode(hand, 0)

        x = agent.convert_to_data_batch(already_played, board, hand, actions)
        self.assertTrue(np.all(x == np.array([
            [
                0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
            ],
            [
                0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
            ]
        ])))

    def test_init_fit_predict_model(self):
        """ Performs basic model operations. """

        already_played = np.array([0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0])
        board = np.array([0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        hand = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        action = np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        agent = DeepQAgent(0)
        agent.start_episode(hand, 0)

        # Fit observation to network by filling replay buffer
        x = agent.convert_to_data_batch(
            already_played, board, hand, [action])
        agent.replay_buffer.add_batch((x, 1.0))
        for _ in range(10):
            agent.fit_values_to_network()

        # Predict for same situation
        pred_q_value = agent.predict_q_values_from_network(
            already_played, board, hand, [action]
        )[0]

        # Prediction should be between -1 and 1
        self.assertLess(pred_q_value, 1.0)
        self.assertGreater(pred_q_value, -1.0)


if __name__ == '__main__':
    unittest.main()
