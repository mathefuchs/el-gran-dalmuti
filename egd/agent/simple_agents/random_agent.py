import numpy as np

from egd.agent.base_agent import ModelBase


class RandomAgent(ModelBase):

    def get_action_values(self, possible_actions: np.ndarray) -> np.ndarray:
        """ Retrieves the values for all provided actions.

        Args:
            possible_actions (np.ndarray): Possible actions

        Returns:
            np.ndarray: Values for all provided actions
        """

        # Return zero vector to decide randomly
        return np.zeros(possible_actions.shape[0], dtype=np.int8)
