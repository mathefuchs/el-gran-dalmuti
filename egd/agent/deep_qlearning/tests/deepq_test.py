import numpy as np
import pandas as pd
import unittest

from egd.game.cards import NUM_CARD_VALUES, get_cards_array
from egd.game.state import random_initial_cards
from egd.agent.state import GameState
from egd.agent.deep_qlearning.deepq_agent import DeepQAgent


class DeepQAgentTest(unittest.TestCase):

    def test_follow_up_actions_batch(self):
        """ Tests the conversion to feed batch. """

        state = GameState()

        state.curr_ap = np.array([0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0])
        state.ap_start = state.curr_ap
        state.curr_board = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        state.board_start = state.curr_board
        hand = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        actions = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        agent = DeepQAgent(0)
        agent.start_episode(hand, state, 0)

        x = agent.follow_up_actions_batch(actions)
        self.assertTrue(np.all(np.isclose(x, np.array([
            [
                0, 0, 0.66666667, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0.33333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5
            ],
            [
                0, 0, 0.66666667, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0.33333333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5
            ]
        ]))))

    def test_init_fit_predict_model(self):
        """ Performs basic model operations. """

        state = GameState()

        state.curr_ap = np.array([0, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0])
        state.ap_start = state.curr_ap
        state.curr_board = np.array([0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        state.board_start = state.curr_board
        hand = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        action = np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        agent = DeepQAgent(0)
        agent.start_episode(hand, state, 0)

        # Fit observation to network by filling replay buffer
        x = agent.follow_up_actions_batch([action])
        pred_before = agent.predict_q_values_from_network(x)[0]
        agent.replay_buffer.add_batch((x, 1.0))
        for _ in range(10):
            agent.fit_values_to_network()

        # Predict for same situation
        pred_q_value = agent.predict_q_values_from_network(x)[0]

        # Prediction should be between 0 and 1 and should converge
        self.assertLess(pred_q_value, 1.0)
        self.assertGreater(pred_q_value, 0.0)
        self.assertLess(pred_before, pred_q_value)


if __name__ == '__main__':
    unittest.main()
