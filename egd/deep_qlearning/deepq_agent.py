import os

# Use GPU for model training makes training
# slower due to small batch size of 1
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"  # nopep8

import numpy as np
import pandas as pd

from keras import models
from keras import layers

from egd.game.cards import NUM_CARD_VALUES
from egd.game.state import has_finished, NUM_PLAYERS
from egd.game.moves import possible_next_moves


class DeepQAgent:

    def __init__(self, playerIndex, create_model=True):
        """ Initialize an agent. """

        self.alpha = 0.5  # learning rate
        self.gamma = 0.95  # favour future rewards
        self.exploration_decay_rate = 1 / 2000
        self.rewards = {
            0: 10.0,  # No other agent finished before
            1: 5.0,  # One other agent finished before
            2: 4.0,  # Two other agents finished before
            3: -10.0,  # Three other agents finished before
        }

        self.playerIndex = playerIndex
        self.debug = False

        # Initialize model
        if create_model:
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

        if self.debug:
            print("Save Trained Deep Q Model.")
        self.network.save("./egd/saved_agents/deepq.h5")

    def load_model(self):
        """ Load model from file. """

        if self.debug:
            print("Load Deep Q Model from file.")
        self.network = models.load_model("./egd/saved_agents/deepq.h5")

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
            epochs=weight, verbose=(1 if self.debug else 0)
        )

    def predict_q_values_from_network(self, already_played, board, hand, actions):
        """ Predicts q-values from the trained neural net. """

        return self.network.predict(
            self.convert_to_data_batch(already_played, board, hand, actions)
        ).flatten()

    def do_step(self, already_played, board, agents_finished,
                next_action_wins_board=lambda a, b: False,
                always_use_best=False, print_luck=False):
        """
            Performs a step in the game.

            Returns (Player finished, Already played cards,
                New board, Best decision made randomly)
        """

        # If player has already finished, pass
        if has_finished(self.hand):
            return True, already_played, board, False

        # Possible actions; Pass if no possible play
        possible_actions = possible_next_moves(self.hand, board)
        if len(possible_actions) == 1 and \
                np.all(possible_actions[0] == 0):
            return False, already_played, board, False

        # Do random decisions with a fixed probability
        best_decision_made_randomly = False
        if not always_use_best and np.random.uniform() < self.epsilon:
            # Chose action randomly
            random_choice = True
            action_index = np.random.choice(len(possible_actions))
            action_taken = possible_actions[action_index]

            # Get q-value estimate only for chosen action
            possible_qvalues = self.predict_q_values_from_network(
                already_played, board, self.hand,
                action_taken
            )[0]
        else:
            # Get predictions for all possible actions
            possible_qvalues = self.predict_q_values_from_network(
                already_played, board, self.hand,
                possible_actions  # Possible actions
            )
            close_to_max = np.isclose(possible_qvalues,
                                      np.nanmax(possible_qvalues))

            # Debug "luck"
            best_decision_made_randomly = np.count_nonzero(close_to_max) > 1
            if print_luck and best_decision_made_randomly:
                print("Player", self.playerIndex,
                      "- Warning: Decision made randomly")

            # Get best action with random tie-breaking
            random_choice = False
            action_index = np.random.choice(np.flatnonzero(close_to_max))
            action_taken = possible_actions[action_index]

        # Compute next state
        next_hand = self.hand - action_taken
        next_board = board if np.all(action_taken == 0) else action_taken
        next_already_played = already_played + action_taken

        # Do not train in inference mode
        if not always_use_best:
            # Retrieve next state's max q-value
            next_possible_actions = \
                possible_next_moves(next_hand, next_board)
            next_qvalues = self.predict_q_values_from_network(
                next_already_played, next_board, next_hand,
                next_possible_actions)
            next_max = np.nanmax(next_qvalues)

            # Determine reward
            if has_finished(next_hand):
                reward_earned = self.rewards[agents_finished]
            elif next_action_wins_board(next_already_played, next_board):
                reward_earned = 1.5
            else:
                reward_earned = 0.0

            # Determine new q-value
            old_qvalue = possible_qvalues[action_index] \
                if not random_choice else possible_qvalues
            new_qvalue = (1 - self.alpha) * old_qvalue + \
                self.alpha * (reward_earned + self.gamma * next_max)

            # Fit neural net to newly observed reward estimate
            self.fit_value_to_network(
                already_played, board, self.hand,
                action_taken, new_qvalue, weight=1
            )

        # Return next state
        self.hand = next_hand
        return (has_finished(self.hand), next_already_played,
                next_board, best_decision_made_randomly)
