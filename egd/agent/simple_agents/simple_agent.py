import numpy as np

from egd.agent.base_agent import ModelBase


class SimpleAgent(ModelBase):

    def decide_action_to_take(
            self, already_played, board, always_use_best,
            print_luck, possible_actions):
        """ Returns (possible_qvalues, action_index, action_taken, 
            random_choice, best_decision_made_randomly) """

        # Either take first action if board empty
        # (no pass move in list) or second action
        action_index = 0 if np.all(board == 0) else 1
        action_taken = possible_actions[action_index]

        return (None, action_index, action_taken, False, False)
