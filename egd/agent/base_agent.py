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
    step_add_replay = 2


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
        """ Returns (possible_qvalues, action_index, action_taken, 
            random_choice, best_decision_made_randomly) """

        pass

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
            always_use_best=False, print_luck=False):
        """
            Performs a (partial) step in the game.

            Returns (Next step state, Player finished, 
                Already played cards, New board, 
                Best decision made randomly)
        """

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
        (possible_qvalues, action_index, action_taken,
         random_choice, best_decision_made_randomly) = \
            self.decide_action_to_take(
                already_played, board, always_use_best,
                print_luck, possible_actions)

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
