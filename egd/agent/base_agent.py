import enum
import abc
import numpy as np
import tensorflow as tf
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.specs import tensor_spec

from egd.config import use_small_nums
from egd.game.cards import NUM_CARD_VALUES
from egd.game.state import has_finished, NUM_PLAYERS
from egd.game.moves import possible_next_moves


class StepState(enum.Enum):
    step_completed = 0
    step_needs_predict = 1


class ModelBase(abc.ABC):

    def __init__(self, playerIndex, debug=False):
        """ Initialize an agent. """

        self.playerIndex = playerIndex
        self.debug = debug

    def start_episode(self, initial_hand, num_episode=0):
        """ Initialize game with assigned initial hand. """

        self.hand = initial_hand
        self.num_episode = num_episode

    def save_model(self):
        """ Save the model to the specified path. """

        pass

    def load_model(self):
        """ Load model from file. """

        pass

    def evaluate_inference_mode(self):
        """ Evaluates q-value representation of 
            neural network in validation games. """

        return None

    def prepare_step(self):
        """ Prepares the step to do. """

        pass

    @abc.abstractmethod
    def decide_action_to_take(
            self, already_played, board, always_use_best,
            print_luck, possible_actions):
        """ Returns (
                decision_made=True, possible_qvalues, action_index, 
                action_taken, random_choice, best_decision_made_randomly
            ) or (
                decision_made=False, random_action_taken or -1,
                inputs to predict
            )
        """

        pass

    def decide_action_based_on_predictions(
            self, predictions_made, print_luck, 
            possible_actions, rand_action_index):
        """ Returns (
                possible_qvalues, action_index, action_taken, 
                random_choice, best_decision_made_randomly
            )
        """

        return ([], -1, None, False, False)

    def process_next_board_state(
            # Last board state
            self, already_played, board,
            # Next board state
            next_already_played, next_board, next_hand,
            # Decided action
            possible_qvalues, action_index, action_taken, random_choice,
            # Other parameters
            agents_finished, next_action_wins_board, always_use_best):
        """ Processes the next board state. """

        pass

    def do_step(
            # Where to resume the step
            self, last_step_state,
            # Board state
            already_played, board, agents_finished,
            # Whether a specified action wins the board
            next_action_wins_board=lambda a, b: False,
            # Other parameters
            always_use_best=False, print_luck=False,
            # Step parameters if last_step_state is step_needs_predict
            required_predictions=None, actions_for_pred=None, 
            rand_action_index=-1):
        """
            Performs a (partial) step in the game.

            Returns 
                (
                    StepState.step_completed, Player finished, 
                    Already played cards, New board, 
                    Best decision made randomly
                ) or (
                    StepState.step_needs_predict, rand_action_index,
                    inputs to predict, Already played cards, board
                )
        """

        # Begin new step if last step completed
        if last_step_state is StepState.step_completed:

            # Prepares the step to do
            self.prepare_step()

            # If player has already finished, pass
            if has_finished(self.hand):
                return (StepState.step_completed, True,
                        already_played, board, False)

            # Possible actions; Pass if no possible play
            possible_actions = possible_next_moves(self.hand, board)
            if len(possible_actions) == 1 and \
                    np.all(possible_actions[0] == 0):
                return (StepState.step_completed, False,
                        already_played, board, False)

            # Decide action to take
            decision_made = self.decide_action_to_take(
                already_played, board, always_use_best,
                print_luck, possible_actions)
            if decision_made[0]:
                (_, possible_qvalues, action_index, action_taken,
                 random_choice, best_decision_made_randomly) = decision_made
            else:
                _, rand_action_index, inputs_to_predict = decision_made
                return (StepState.step_needs_predict, rand_action_index,
                        inputs_to_predict, already_played, board)

        # Process computed q-values if available.
        if last_step_state is StepState.step_needs_predict:
            # Process predictions received
            (possible_qvalues, action_index, action_taken,
             random_choice, best_decision_made_randomly) = \
                self.decide_action_based_on_predictions(
                    required_predictions, print_luck, 
                    actions_for_pred, rand_action_index)

        # Compute next state
        next_hand = self.hand - action_taken
        next_board = board if np.all(action_taken == 0) else action_taken
        next_already_played = already_played + action_taken

        # Process next state
        self.process_next_board_state(
            already_played, board,
            next_already_played, next_board, next_hand,
            possible_qvalues, action_index, action_taken, random_choice,
            agents_finished, next_action_wins_board, always_use_best
        )

        # Return next state
        self.hand = next_hand
        return (StepState.step_completed, has_finished(self.hand),
                next_already_played, next_board, best_decision_made_randomly)
