import numpy as np

from egd.agent.base_agent import ModelBase


class RandomAgent(ModelBase):

    def decide_action_to_take(
            self, already_played, board, always_use_best,
            print_luck, possible_actions):
        """ Returns (possible_qvalues, action_index, action_taken, 
            random_choice, best_decision_made_randomly) """

        # Decide randomly
        action_index = np.random.randint(len(possible_actions))
        action_taken = possible_actions[action_index]

        return (True, None, action_index, action_taken, True, True)
