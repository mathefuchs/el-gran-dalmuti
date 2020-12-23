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


class ModelBase(abc.ABC):

    def __init__(self, playerIndex, debug=False):
        """ Initialize an agent. """

        self.trainable = False  # Whether agent is trainable
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
            # Possible states before the next move of this agent
            list_next_possible_states, next_ap, next_b, next_hand,
            # Decided action
            possible_qvalues, action_index, action_taken, random_choice,
            # Other parameters
            agents_finished, always_use_best):
        """ Processes the next board state. """

        pass

    def do_step(
            # Board state
            self, already_played, board, agents_finished,
            # Possible states before the next move of this agent
            list_next_possible_states=lambda ap, b: ([], []),
            # Other parameters
            always_use_best=False, print_luck=False):
        """
            Performs a (partial) step in the game.

            Returns (Player finished, 
                Already played cards, New board, 
                Best decision made randomly)
        """

        # Prepares the step to do
        self.prepare_step()

        # If player has already finished, pass
        if has_finished(self.hand):
            return True, already_played, board, False

        # Possible actions; Pass if no possible play
        possible_actions = possible_next_moves(self.hand, board)
        if len(possible_actions) == 1 and \
                np.all(possible_actions[0] == 0):
            return False, already_played, board, False

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
            already_played, board, list_next_possible_states,
            next_already_played, next_board, next_hand, possible_qvalues,
            action_index, action_taken, random_choice, agents_finished,
            always_use_best
        )

        # Return next state
        self.hand = next_hand
        return (has_finished(self.hand), next_already_played,
                next_board, best_decision_made_randomly)
