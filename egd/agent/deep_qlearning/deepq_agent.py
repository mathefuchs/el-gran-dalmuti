import numpy as np
import tensorflow as tf
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.specs import tensor_spec

from egd.agent.base_agent import ModelBase
from egd.config import use_small_nums
from egd.game.cards import NUM_CARD_VALUES
from egd.game.state import has_finished, NUM_PLAYERS
from egd.game.moves import possible_next_moves


class DeepQAgent(ModelBase):

    def __init__(self, playerIndex, debug=False, create_model=True):
        """ Initialize an agent. """

        super().__init__(playerIndex, debug=debug)

        # Whether to use small numbers for debugging reasons
        self.use_small_numbers = use_small_nums

        # Hyperparameters
        self.alpha = 0.75  # learning rate
        self.gamma = 0.95  # favour future rewards
        self.exploration_decay_rate = 1 / 2000
        self.reward_win_round = 0.005
        self.reward_per_card_played = 0.001
        self.rewards = {
            0: 1.0,  # No other agent finished before
            1: 0.05,  # One other agent finished before
            2: 0.04,  # Two other agents finished before
            3: -1.0,  # Three other agents finished before
        }

        # Training/Batch parameters
        self.sample_batch = 64 if self.use_small_numbers else 512
        self.replay_capacity = 128 if self.use_small_numbers else 1024
        self.train_each_n_steps = 64 if self.use_small_numbers else 512
        self.step_iteration = 0
        self.model_data_spec = (
            tf.TensorSpec([4 * 13], tf.int8, "board_state"),
            tf.TensorSpec([1], tf.float32, "q_value"),
        )
        self.replay_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
            capacity=self.replay_capacity,
            data_spec=tensor_spec.to_nest_array_spec(self.model_data_spec)
        )

        # Validation parameters
        self.val_replay_capacity = 20 if self.use_small_numbers else 200
        self.validation_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
            capacity=self.val_replay_capacity,
            data_spec=tensor_spec.to_nest_array_spec(self.model_data_spec)
        )

        # Initialize model
        if create_model:
            self._create_model()

    def _create_model(self):
        """ Create model for predicting q-values. """

        # Create sequential model
        self.network = tf.keras.Sequential()

        # Use simple fully-connected layers
        self.network.add(tf.keras.layers.Dense(
            4 * 13,
            activation=tf.keras.activations.relu
        ))
        self.network.add(tf.keras.layers.Dense(
            13,
            activation=tf.keras.activations.relu
        ))

        # Final dense layer for q-value
        self.network.add(tf.keras.layers.Dense(1))

        # Compile neural network, use mean-squared error
        self.network.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.SGD(),
            metrics=[
                tf.keras.losses.Huber(),
                tf.keras.losses.MeanSquaredError(),
            ]
        )

    def start_episode(self, initial_hand, num_episode=0):
        """ Initialize game with assigned initial hand. """

        super().start_episode(initial_hand, num_episode)

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
        self.network = tf.keras.models.load_model(
            "./egd/saved_agents/deepq.h5")

    def convert_to_data_batch(self, already_played, board, hand, actions):
        """ Converts the given arrays to a representation 
            understood by the model. """

        stack_list = []
        for action in actions:
            if action.shape[0] != NUM_CARD_VALUES:
                raise Exception("Action has wrong shape.")

            stack_list.append(np.hstack([
                already_played, board, hand, action
            ]))

        return np.vstack(stack_list)

    @tf.autograph.experimental.do_not_convert
    def evaluate_inference_mode(self):
        """ Evaluates q-value representation of 
            neural network in validation games. """

        dataset = self.validation_buffer.as_dataset(
            sample_batch_size=self.val_replay_capacity)

        return self.network.evaluate(
            dataset, steps=1, verbose=1
        )

    @tf.autograph.experimental.do_not_convert
    def fit_values_to_network(self):
        """ Fits data from the replay buffer to the neural net. """

        dataset = self.replay_buffer.as_dataset(
            sample_batch_size=self.sample_batch)

        self.network.fit(
            dataset, steps_per_epoch=1, epochs=1, verbose=1
        )

    def predict_q_values_from_network(
            self, already_played, board, hand, actions):
        """ Predicts q-values from the trained neural net. """

        return self.network.predict(
            self.convert_to_data_batch(already_played, board, hand, actions)
        ).flatten()

    def decide_action_to_take(
            self, already_played, board, always_use_best,
            print_luck, possible_actions):
        """ Returns (possible_qvalues, action_index, action_taken, 
            random_choice, best_decision_made_randomly) """

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
                [action_taken]
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

        return (possible_qvalues, action_index, action_taken,
                random_choice, best_decision_made_randomly)

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

        # Retrieve next state's max q-value
        next_possible_actions = \
            possible_next_moves(next_hand, next_board)
        next_qvalues = self.predict_q_values_from_network(
            next_already_played, next_board, next_hand,
            next_possible_actions)
        next_max = np.nanmax(next_qvalues)

        # Determine reward
        if has_finished(next_hand):
            # Reward based on how many other agents are already finished
            reward_earned = self.rewards[agents_finished]
        elif next_action_wins_board(next_already_played, next_board):
            # Cards that win a round safely gain fixed rewards
            reward_earned = self.reward_win_round
        else:
            # Else, the more cards played the better
            reward_earned = self.reward_per_card_played * \
                np.linalg.norm(action_taken, 1)

        # Determine new q-value
        old_qvalue = possible_qvalues[action_index] \
            if not random_choice else possible_qvalues
        new_qvalue = (1 - self.alpha) * old_qvalue + \
            self.alpha * (reward_earned + self.gamma * next_max)

        # Do not train in inference mode
        if not always_use_best:
            # Record step in replay buffer
            self.replay_buffer.add_batch((
                self.convert_to_data_batch(
                    already_played, board, self.hand, [action_taken]
                ), new_qvalue))

            # Fit neural net to observed replays
            if self.step_iteration != 0 and self.step_iteration % \
                    self.train_each_n_steps == 0:
                self.fit_values_to_network()
            self.step_iteration += 1

        # Validate q-values in inference mode
        else:
            self.validation_buffer.add_batch((
                self.convert_to_data_batch(
                    already_played, board, self.hand, [action_taken]
                ), new_qvalue))
