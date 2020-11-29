import os

# Use GPU for model training
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"  # nopep8

import numpy as np
import pandas as pd

from keras import models
from keras import layers

from egd.game.state import has_finished, NUM_PLAYERS
from egd.game.moves import possible_next_moves


class DeepQAgent:

    def __init__(self, playerIndex):
        """ Initialize an agent. """

        self.alpha = 0.5  # learning rate
        self.gamma = 0.95  # favour future rewards
        self.exploration_decay_rate = 1 / 2000
        self.rewards = {
            0: 1.0,  # No other agent finished before
            1: 0.6,  # One other agent finished before
            2: 0.5,  # Two other agents finished before
            3: -1.0,  # Three other agents finished before
        }

        self.playerIndex = playerIndex
        self.debug = False

        # Initialize model
        self._create_model()

    def _create_model(self):
        """ Create model for predicting q-values. """

        # Create sequential model
        self.network = models.Sequential()

        # Use to convolutional layers to group same cards
        # of a kind among the different input vectors
        self.network.add(layers.convolutional.Conv2D(
            # Input rows: Already played, board, hand, action
            input_shape=(4, 13, 1),
            # Window size
            kernel_size=(3, 3),
            # Output filters
            filters=11,
            # Activation
            activation='relu'
        ))
        self.network.add(layers.convolutional.Conv2D(
            # Window size
            kernel_size=(2, 2),
            # Output filters
            filters=13,
            # Activation
            activation='relu'
        ))

        # Flatten convolutional layers
        self.network.add(layers.Flatten())
        self.network.add(layers.Dense(13, activation="relu"))

        # Dropout to remove noise
        self.network.add(layers.Dropout(0.5))

        # Final dense layer for q-value
        self.network.add(layers.Dense(1))

        # Compile neural network, use mean-squared error
        self.network.compile(
            loss='mse', optimizer='RMSprop', metrics=['mse']
        )

    def start_episode(self, initial_hand, num_episode=0):
        """ Initialize game with assigned initial hand. """

        self.hand = initial_hand
        self.num_episode = num_episode
        # amount of random decisions
        self.epsilon = 1 / np.sqrt(
            num_episode * self.exploration_decay_rate + 1)

    def save_model(self):
        """ Save the model to the specified path. """

        # TODO Implement saving
        pass

    def load_model(self):
        """ Load model from file. """

        # TODO Implement loading
        pass

    def convert_to_data_batch(self, already_played, board, hand, actions):
        """ Converts the given arrays to a representation understood by the model. """

        enc_already_played = np.resize(already_played, (13, 1))
        enc_board = np.resize(board, (13, 1))
        enc_hand = np.resize(hand, (13, 1))

        stack_list = []
        for action in actions:
            stack_list.append([
                enc_already_played, enc_board,
                enc_hand, np.resize(action, (13, 1))
            ])

        return np.stack(stack_list, axis=0)

    def fit_value_to_network(self, already_played, board, hand, action,
                             updated_q_value, weight=1):
        """ Fits a measured q-value to the neural net. """

        self.network.fit(
            self.convert_to_data_batch(already_played, board, hand, [action]),
            np.array([[updated_q_value]]),
            epochs=weight
        )

    def predict_q_values_from_network(self, already_played, board, hand, actions):
        """ Predicts q-values from the trained neural net. """

        return self.network.predict(
            self.convert_to_data_batch(already_played, board, hand, actions)
        ).flatten()

    def do_step(self, already_played, board, agents_finished,
                always_use_best=False, print_luck=False):
        """
            Performs a step in the game.

            Returns (Player finished, Already played cards, New board)
        """

        # If player has already finished, pass
        if has_finished(self.hand):
            return True, already_played, board

        # Possible actions; Pass if no possible play
        possible_hands, possible_boards = \
            possible_next_moves(self.hand, board)
        if len(possible_hands) == 1 and \
                np.all(possible_boards[0] == board):
            return False, already_played, board

        # Do random decisions with a fixed probability
        if not always_use_best and np.random.uniform() < self.epsilon:
            action_index = np.random.choice(len(possible_hands))
        else:
            # Get predictions for all possible actions
            possible_qvalues = self.predict_q_values_from_network(
                already_played, board, self.hand,
                self.hand - possible_hands  # Possible actions
            )

            # Get best action with random tie-breaking
            action_index = np.random.choice(
                np.flatnonzero(np.isclose(
                    possible_qvalues, np.nanmax(possible_qvalues)
                )))

        # Compute next state
        next_hand = possible_hands[action_index]
        next_board = possible_boards[action_index]
        next_already_played = already_played + next_board \
            if not np.all(next_board == board) else already_played

        # TODO

        # Return next state
        self.hand = next_hand
        return has_finished(self.hand), next_already_played, next_board
