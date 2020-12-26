import abc
import numpy as np
from scipy.special import softmax
from egd.agent.state import GameState, StepOptions
from egd.game.state import has_finished
from egd.game.moves import possible_next_moves, only_passing


class ModelBase(abc.ABC):

    def __init__(self, playerIndex: int, debug=False):
        """ Initialize an agent

        Args:
            playerIndex (int): Player's index
            debug (bool, optional): Debug flag. Defaults to False.
        """

        self.trainable = False  # Whether agent is trainable
        self.playerIndex = playerIndex
        self.debug = debug

    def save_model(self):
        """ Save the model to the specified path. """

        pass

    def load_model(self):
        """ Load model from file. """

        pass

    def start_episode(
            self, initial_hand: np.ndarray,
            state: GameState, num_episode=0):
        """ Initialize game with assigned initial hand

        Args:
            initial_hand (np.ndarray): Initial hand
            state (GameState): Current game state
            num_episode (int, optional): Num of epochs. Defaults to 0.
        """

        self.hand = initial_hand
        self.state = state
        self.num_episode = num_episode

    def evaluate_inference_mode(self) -> list:
        """ Evaluates performance in validation games.

        Returns:
            list: Loss metrics of evaluation
        """

        pass

    def prepare_step(self):
        """ Prepares the step to do. """

        pass

    @abc.abstractmethod
    def get_action_values(self, possible_actions: np.ndarray) -> np.ndarray:
        """ Retrieves the values for all provided actions.

        Args:
            possible_actions (np.ndarray): Possible actions

        Returns:
            np.ndarray: Values for all provided actions
        """

        pass

    def process_next_board_state(
            self, action_values: np.ndarray, action_probabilities: np.ndarray,
            action_index: int, action_taken: np.ndarray, options: StepOptions):
        """ Processes the next board state. """

        pass

    def do_step(self, options: StepOptions) -> bool:
        """ Performs a step in the game.

        Args:
            options (StepOptions): Step options

        Returns:
            bool: Decision made randomly
        """

        # Prepares the step to do
        self.prepare_step()

        # If player has already finished, pass
        if has_finished(self.hand):
            self.state.report_empty_action()
            self.state.report_agent_finished()
            return False

        # Possible actions; Pass if no possible play
        possible_actions = possible_next_moves(
            self.hand, self.state.curr_board)
        if only_passing(possible_actions):
            self.state.report_empty_action()
            return False

        # Decide action to take
        action_values = self.get_action_values(possible_actions)
        action_probabilities = softmax(action_values)

        # Sample action according to probabilities
        if self.trainable and not options.inference_mode:
            decision_made_randomly = True
            action_index = np.random.choice(
                len(possible_actions), p=action_probabilities)
            action_taken = possible_actions[action_index]

        # Else, choose action with max probability in inference
        else:
            close_to_max = np.isclose(
                action_probabilities, np.max(action_probabilities))

            # Log ties in inference mode decisions
            decision_made_randomly = np.count_nonzero(close_to_max) > 1
            if options.print_tie_in_inference and decision_made_randomly:
                print("Player", self.playerIndex,
                      "- Warning: Decision made randomly")

            # Choose best decision
            action_index = np.random.choice(np.flatnonzero(close_to_max))
            action_taken = possible_actions[action_index]

        # Compute next state
        next_hand = self.hand - action_taken
        next_board = self.state.curr_board \
            if np.all(action_taken == 0) else action_taken
        next_ap = self.state.curr_ap + action_taken

        # Process next state
        self.process_next_board_state(
            action_values, action_probabilities,
            action_index, action_taken, options)

        # Update state
        self.hand = next_hand
        self.state.curr_ap = next_ap
        self.state.curr_board = next_board
        self.state.report_action(action_taken)

        # Set start state after the action of the learning agent
        if self.trainable:
            self.state.ap_start = next_ap
            self.state.board_start = next_board

        # Report that agent finished if applicable
        if has_finished(self.hand):
            self.state.report_agent_finished()

        # Return whether decision was made randomly
        return decision_made_randomly
