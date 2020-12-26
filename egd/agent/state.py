import numpy as np

from egd.game.cards import NUM_CARD_VALUES, EMPTY_HAND
from egd.game.state import NUM_PLAYERS, PLAYERS


class GameState:

    def __init__(self):
        """ Initializes a state object. """

        # Past actions
        self.past_actions = np.zeros(
            (NUM_PLAYERS, NUM_CARD_VALUES), dtype=np.int8)

        # Current state
        self.ap_start = EMPTY_HAND
        self.board_start = EMPTY_HAND
        self.curr_ap = EMPTY_HAND
        self.curr_board = EMPTY_HAND

        # Ranking
        self.agents_finished = np.zeros(NUM_PLAYERS, dtype=np.bool)
        self.agent_ranking = []

        # Order of play
        self.order_of_play = np.random.permutation(PLAYERS)
        self.current_player_index = np.random.randint(NUM_PLAYERS)
        self.turns_passed_without_move = 0

    def current_player(self) -> int:
        """ Returns the current agent's index.

        Returns:
            int: The agent whose turn it is.
        """

        return self.order_of_play[self.current_player_index]

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

    def all_agents_finished(self) -> bool:
        """ Whether all agents have finished.

        Returns:
            bool: Whether all agents have finished.
        """

        return self.get_num_agents_finished() == NUM_PLAYERS

    def last_action_passing(self) -> bool:
        """ Whether the last action was passing.

        Returns:
            bool: Whether the last action was passing.
        """

        return np.all(self.past_actions[-1] == 0)

    def next_turn(self):
        """ Initiate next turn. """

        self.current_player_index = (
            self.current_player_index + 1) % NUM_PLAYERS

    def check_all_passed(self) -> bool:
        """ Reset board after a complete round of no moves.

        Returns:
            bool: Returns whether the board state has 
            been reset due to agents passing.
        """

        if self.last_action_passing():
            self.turns_passed_without_move += 1

            if self.turns_passed_without_move == NUM_PLAYERS - 1:
                # Reset board
                self.turns_passed_without_move = 0
                self.board_start = EMPTY_HAND
                self.curr_board = EMPTY_HAND
                return True
        else:
            self.turns_passed_without_move = 0

        return False

    def next_agent_passing_leads_to_reset(self) -> bool:
        """ Whether passing of the next agent 
        leads to the reset of the board.

        Returns:
            bool: Whether passing leads to reset.
        """

        return self.turns_passed_without_move == NUM_PLAYERS - 2


class EvaluationStats:

    def __init__(self):

        # Statistics to evaluate
        self.number_decisions = np.zeros(NUM_PLAYERS, dtype=np.int16)
        self.decisions_randomly = np.zeros(NUM_PLAYERS, dtype=np.int16)

    def log_random_decision(self, current_player: int, random=False):
        """ Logs whether a decision was made randomly.

        Args:
            current_player (int): Player to log the decision for.
            random (bool, optional): Whether the 
            decision was made randomly. Defaults to False.
        """

        self.number_decisions[current_player] += 1
        self.decisions_randomly[current_player] += 1 if random else 0

    def get_amount_of_random_decisions(self) -> np.ndarray:
        """ Retrieves the amount of random decision for all agents.

        Returns:
            np.ndarray: The amount of random decisions.
        """

        return self.decisions_randomly / self.number_decisions


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
