import numpy as np

from egd.agent.base_agent import ModelBase


class SimpleAgent(ModelBase):

    def get_action_values(self, possible_actions: np.ndarray) -> np.ndarray:
        """ Retrieves the values for all provided actions.

        Args:
            possible_actions (np.ndarray): Possible actions

        Returns:
            np.ndarray: Values for all provided actions
        """

        # Either take first action (lowest value cards) if board
        # empty (no pass move in list) or second action
        action_index = 0 if np.all(self.state.curr_board == 0) else 1

        # Return unit vector with choice
        action_values = np.zeros(possible_actions.shape[0], dtype=np.int8)
        action_values[action_index] = 1
        return action_values
