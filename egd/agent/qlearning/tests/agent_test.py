import numpy as np
import pandas as pd
import unittest

from egd.game.cards import NUM_CARD_VALUES, get_cards_array
from egd.game.state import random_initial_cards
from egd.agent.base_agent import StepState
from egd.agent.qlearning.qtable import QTable
from egd.agent.qlearning.qagent import QLearningAgent


class QLearningAgentTest(unittest.TestCase):

    def test_perform_step_beginning_of_game(self):
        """ Performs step and checks state. """

        initial_cards = random_initial_cards()

        agent = QLearningAgent(0)
        agent.debug = False
        agent.start_episode(initial_cards[0], 0)

        _, finished, new_already_played, new_board, _ = \
            agent.do_step(StepState.step_completed,
                          np.zeros(NUM_CARD_VALUES, dtype=np.int8),
                          np.zeros(NUM_CARD_VALUES, dtype=np.int8),
                          0, always_use_best=True)  # Enforce deterministic behaviour

        self.assertFalse(finished)
        self.assertFalse(np.all(new_already_played == 0))
        self.assertFalse(np.all(new_board == 0))

    def test_perform_final_step(self):
        """ Performs final step and checks state. """

        agent = QLearningAgent(0)
        agent.debug = False
        hand = get_cards_array(2, 1)
        agent.start_episode(hand, 0)

        already_played = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
        board = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
        _, finished, new_already_played, new_board, _ = \
            agent.do_step(StepState.step_completed,
                          already_played, board, 0)

        self.assertTrue(finished)
        self.assertFalse(np.all(new_already_played == 0))
        self.assertFalse(np.all(new_board == 0))
        self.assertTrue(np.all(agent.hand == 0))
        self.assertAlmostEqual(agent.qtable.get_qtable_entry(
            already_played, board, hand).iloc[0, 0], agent.alpha)

    def test_agent_has_to_pass(self):
        """ 
            Agent has to pass his turn to the next 
            player without playing any cards. 
        """

        agent = QLearningAgent(0)
        agent.debug = False
        hand = get_cards_array(5, 2)
        agent.start_episode(hand, 0)

        already_played = get_cards_array(2, 2)
        board = get_cards_array(2, 2)
        _, finished, new_already_played, new_board, _ = \
            agent.do_step(StepState.step_completed,
                          already_played, board, 0)

        self.assertFalse(finished)
        self.assertTrue(np.all(new_already_played == already_played))
        self.assertTrue(np.all(new_board == board))
        self.assertTrue(np.all(agent.hand == hand))


if __name__ == '__main__':
    unittest.main()
