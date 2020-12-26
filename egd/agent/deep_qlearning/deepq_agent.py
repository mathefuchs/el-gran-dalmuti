import numpy as np
import tensorflow as tf
from typing import List
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.specs import tensor_spec

from egd.config import use_small_nums
from egd.agent.state import GameState, StepOptions
from egd.agent.base_agent import ModelBase
from egd.game.cards import NUM_CARD_VALUES, AVAILABLE_CARDS, EMPTY_HAND
from egd.game.state import has_finished, NUM_PLAYERS
from egd.game.moves import possible_next_moves


class DeepQAgent(ModelBase):

    def __init__(self, playerIndex: int, debug=False, create_model=True):
        """ Initialize an agent.

        Args:
            playerIndex (int): Player's index
            debug (bool, optional): Debug flag. Defaults to False.
            create_model (bool, optional): Whether to 
            create model. Defaults to True.
        """

        super().__init__(playerIndex, debug=debug)
        self.trainable = True

        # Whether to use small numbers for debugging reasons
        self.use_small_numbers = use_small_nums

        # Hyperparameters
        self.alpha = 0.01  # learning rate
        self.gamma = 0.95  # favour future rewards
        self.exploration_decay_rate = 1 / 20000
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
        self.train_each_n_steps = 5 if self.use_small_numbers else 50
        self.step_iteration = 0
        self.model_data_spec = (
            tf.TensorSpec([6 * 13], tf.float32, "state_and_actions"),
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
            self.create_model()

    def create_model(self):
        """ Create model for predicting q-values. """

        # Create sequential model
        self.network = tf.keras.Sequential()

        # Use simple fully-connected layers
        self.network.add(tf.keras.layers.Dense(6 * 13, activation="relu"))
        self.network.add(tf.keras.layers.Dense(13, activation="relu"))

        # Final dense layer for q-value
        self.network.add(tf.keras.layers.Dense(1))

        # Compile neural network, use mean-squared error
        self.network.compile(loss="mse", optimizer="RMSprop", metrics=["mse"])

    def start_episode(
            self, initial_hand: np.ndarray,
            state: GameState, num_episode=0):
        """ Initialize game with assigned initial hand

        Args:
            initial_hand (np.ndarray): Initial hand
            state (GameState): Current game state
            num_episode (int, optional): Num of epochs. Defaults to 0.
        """

        super().start_episode(initial_hand, state, num_episode)

        # Amount of random decisions; the larger, the more random
        self.epsilon = max(0.001, 8.0 - num_episode *
                           self.exploration_decay_rate)

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

    def follow_up_actions_batch(self, actions: np.ndarray) -> np.ndarray:
        """ Converts the given actions to a representation 
        understood by the model.

        Args:
            actions (np.ndarray): The possible actions of
            this agent to compute q-values for.

        Raises:
            Exception: If actions has the wrong shape.

        Returns:
            np.ndarray: Data batch to feed to the neural network.
        """

        stack_list = []
        for action in actions:
            if action.shape[0] != NUM_CARD_VALUES:
                raise Exception("Action has wrong shape.")

            # Scale inputs with AVAILABLE_CARDS
            # to represent share of all cards
            stack_list.append(np.hstack([
                self.state.ap_start / AVAILABLE_CARDS,
                self.state.board_start / AVAILABLE_CARDS,
                self.state.past_actions[-3] / AVAILABLE_CARDS,
                self.state.past_actions[-2] / AVAILABLE_CARDS,
                self.state.past_actions[-1] / AVAILABLE_CARDS,
                action / AVAILABLE_CARDS
            ]))

        return np.vstack(stack_list)

    @tf.autograph.experimental.do_not_convert
    def evaluate_inference_mode(self) -> List[float]:
        """ Evaluates q-value representation of 
        neural network in validation games.

        Returns:
            List[float]: Loss metrics of evaluation
        """

        dataset = self.validation_buffer.as_dataset(
            sample_batch_size=self.val_replay_capacity)

        return self.network.evaluate(
            dataset, steps=1, verbose=1)

    @tf.autograph.experimental.do_not_convert
    def fit_values_to_network(self):
        """ Fits data from the replay buffer to the neural net. """

        dataset = self.replay_buffer.as_dataset(
            sample_batch_size=self.sample_batch)

        self.network.fit(
            dataset, steps_per_epoch=1, epochs=1, verbose=1)

    def predict_q_values_from_network(
            self, data_to_predict: np.ndarray) -> np.ndarray:
        """ Predicts q-values from the trained neural net.

        Args:
            data_to_predict (np.ndarray): Data batch to predict.

        Returns:
            np.ndarray: The predictions.
        """

        return self.network.predict(data_to_predict).flatten()

    def get_action_values(self, possible_actions: np.ndarray) -> np.ndarray:
        """ Retrieves the values for all provided actions.

        Args:
            possible_actions (np.ndarray): Possible actions

        Returns:
            np.ndarray: Values for all provided actions
        """

        # Get predictions for all possible actions
        return self.predict_q_values_from_network(
            self.follow_up_actions_batch(possible_actions)) / self.epsilon

    def next_board_after_action(
            self, steps_already_passed: int, prev_board: np.ndarray,
            action_taken: np.ndarray) -> np.ndarray:
        """ Next board after the given action.

        Args:
            steps_already_passed (int): Steps already passed before.
            prev_board (np.ndarray): Previous board.
            action_taken (np.ndarray): Action taken.

        Returns:
            np.ndarray: New board state.
        """

        if np.all(action_taken == 0):
            if steps_already_passed == NUM_PLAYERS - 1:
                return EMPTY_HAND
            else:
                return prev_board
        else:
            return action_taken

    def minimax_qvalue(
            self, next_ap: np.ndarray, next_board: np.ndarray,
            next_hand: np.ndarray, action_taken: np.ndarray,
            next_turns_passed_without_move: int) -> float:
        """ Retrieve the next state's q-value 
        according to the minimax approach.

        Args:
            next_ap (np.ndarray): Next already played.
            next_board (np.ndarray): Next board.
            next_hand (np.ndarray): Next hand.
            action_taken (np.ndarray): Last action taken.

        Returns:
            float: Minimax q-value of next state.
        """

        # Gather all possible action combinations for the next turn
        possible_comb = []
        remaining_cards = AVAILABLE_CARDS - next_ap - next_hand

        # Agents finished
        a1_finished = self.state.agents_finished[self.state.next_player(1)]
        a2_finished = self.state.agents_finished[self.state.next_player(2)]
        a3_finished = self.state.agents_finished[self.state.next_player(3)]
        a4_finished = self.state.agents_finished[self.state.next_player(4)]

        # Iterate over all possible agent 1 moves
        a1_moves = possible_next_moves(remaining_cards, next_board) \
            if not a1_finished else [EMPTY_HAND]
        for a1_move in a1_moves:
            # State after possible move of agent 1
            remaining_cards_a1 = remaining_cards - a1_move
            steps_already_passed_a1 = next_turns_passed_without_move + \
                (1 if np.all(a1_move == 0) else 0)
            board_a1 = self.next_board_after_action(
                steps_already_passed_a1, next_board, a1_move)
            steps_already_passed_a1 %= (NUM_PLAYERS - 1)

            # Iterate over all possible agent 2 moves
            a2_moves = possible_next_moves(remaining_cards_a1, board_a1) \
                if not a2_finished else [EMPTY_HAND]
            for a2_move in a2_moves:
                # State after possible move of agent 2
                remaining_cards_a2 = remaining_cards_a1 - a2_move
                steps_already_passed_a2 = steps_already_passed_a1 + \
                    (1 if np.all(a2_move == 0) else 0)
                board_a2 = self.next_board_after_action(
                    steps_already_passed_a2, board_a1, a2_move)
                steps_already_passed_a2 %= (NUM_PLAYERS - 1)

                # Iterate over all possible agent 3 moves
                a3_moves = possible_next_moves(remaining_cards_a2, board_a2) \
                    if not a3_finished else [EMPTY_HAND]
                for a3_move in a3_moves:
                    # State after possible move of agent 3
                    steps_already_passed_a3 = steps_already_passed_a2 + \
                        (1 if np.all(a3_move == 0) else 0)
                    board_a3 = self.next_board_after_action(
                        steps_already_passed_a3, board_a2, a3_move)

                    # Iterate over all possible moves of this agent
                    a4_moves = possible_next_moves(next_hand, board_a3) \
                        if not a4_finished else [EMPTY_HAND]
                    for a4_move in a4_moves:
                        # Append possible action history
                        possible_comb.append(np.hstack([
                            next_ap / AVAILABLE_CARDS,
                            next_board / AVAILABLE_CARDS,
                            a1_move / AVAILABLE_CARDS,
                            a2_move / AVAILABLE_CARDS,
                            a3_move / AVAILABLE_CARDS,
                            a4_move / AVAILABLE_CARDS
                        ]))

        # Predict q-value of all possible combinations
        possible_comb = np.vstack(possible_comb)
        predicted_qvalues = self.predict_q_values_from_network(possible_comb)

        # Get minimum q-value for opponent actions
        min_qvalue_opponents_index = np.argmin(predicted_qvalues)
        min_qvalue_prefix = possible_comb[
            min_qvalue_opponents_index, :(5 * NUM_CARD_VALUES)]

        # Get maximum q-value for the minimal opponent actions
        minimax_qvalue = np.max(predicted_qvalues[np.all(
            possible_comb[:, :(5 * NUM_CARD_VALUES)] == min_qvalue_prefix, axis=1)])

        # Return minimax value
        return minimax_qvalue

    def process_next_board_state(
            self, next_ap: np.ndarray, next_board: np.ndarray,
            next_hand: np.ndarray, action_taken: np.ndarray,
            next_turns_passed_without_move: int, options: StepOptions):
        """ Processes the next board state.

        Args:
            next_ap (np.ndarray): Next already played state.
            next_board (np.ndarray): Next board.
            next_hand (np.ndarray): Next hand.
            action_taken (np.ndarray): Taken action.
            next_turns_passed_without_move (int): 
            Turns past before next action.
            options (StepOptions): Options.
        """

        # Terminal states have fixed q-values
        if not has_finished(next_hand):
            # Use minimax approach to get next state's q-value
            next_state_value = self.minimax_qvalue(
                next_ap, next_board, next_hand, action_taken,
                next_turns_passed_without_move)

        # Determine reward
        if has_finished(next_hand):
            # Reward based on how many other agents are already finished
            reward_earned = self.rewards[self.state.get_num_agents_finished()]
            # Terminal state has q-value zero
            next_state_value = 0
        else:
            # Else, the more cards played the better
            reward_earned = self.reward_per_card_played * \
                np.linalg.norm(action_taken, 1)

        # Determine new q-value and states
        next_qvalue = reward_earned + self.gamma * next_state_value

        # Do not train in inference mode
        if not options.inference_mode:
            # Record step in replay buffer
            self.replay_buffer.add_batch((
                self.follow_up_actions_batch([action_taken]), next_qvalue))

            # Fit neural net to observed replays
            if (self.step_iteration != 0 and
                    self.step_iteration % self.train_each_n_steps == 0):
                self.fit_values_to_network()
            self.step_iteration += 1

        # Validate q-values in inference mode
        else:
            self.validation_buffer.add_batch((
                self.follow_up_actions_batch([action_taken]), next_qvalue))
