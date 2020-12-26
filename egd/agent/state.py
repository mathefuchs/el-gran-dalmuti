import numpy as np

from egd.game.cards import NUM_CARD_VALUES, EMPTY_HAND
from egd.game.state import NUM_PLAYERS


class GameState:

    def __init__(self):
        """ Initializes a state object. """

        self.past_actions = np.zeros(
            (NUM_PLAYERS, NUM_CARD_VALUES), dtype=np.int8)
        self.ap_start = EMPTY_HAND
        self.board_start = EMPTY_HAND
        self.curr_ap = EMPTY_HAND
        self.curr_board = EMPTY_HAND
        self.agents_finished = np.zeros(NUM_PLAYERS, dtype=np.bool)
        self.agent_ranking = []

    def report_action(self, action: np.ndarray):
        """ Appends action to past actions.

        Args:
            action (np.ndarray): Action to append
        """

        self.past_actions[:-1] = self.past_actions[1:]
        self.past_actions[-1] = action

    def report_empty_action(self):
        """ Report a passing action. """

        self.report_action(EMPTY_HAND)

    def report_agent_finished(self, agent: int):
        """ Report that an agent has finished.

        Args:
            agent (int): The agent that has finished
        """

        # Append agent if newly finished to ranking
        if not self.agents_finished[agent]:
            self.agent_ranking.append(agent)

        self.agents_finished[agent] = True

    def get_num_agents_finished(self) -> int:
        """ Returns the number of finished agents.

        Returns:
            int: Number of finished agents.
        """

        return np.count_nonzero(self.agents_finished)


class StepOptions:

    def __init__(self, print_tie_in_inference=False, inference_mode=False):
        """ Initializes a step options object.

        Args:
            print_tie_in_inference (bool, optional): 
            Whether to print a warning if a tie happens 
            in decision making in inference mode.
            inference_mode (bool, optional): 
            Whether to use best decision.
        """

        self.print_tie_in_inference = print_tie_in_inference
        self.inference_mode = inference_mode


INFERENCE_OPTIONS = StepOptions(False, True)
TRAIN_OPTIONS = StepOptions(False, False)
