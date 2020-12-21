import numpy as np
import pandas as pd

from egd.game.state import has_finished, NUM_PLAYERS
from egd.game.moves import possible_next_moves
from egd.agent.base_agent import ModelBase
from egd.agent.qlearning.qtable import QTable


class QLearningAgent(ModelBase):

    def __init__(self, playerIndex, debug=False):
        """ Initialize an agent. """

        # Initialize base class
        super().__init__(playerIndex, debug=debug)

        # Hyperparameters
        self.alpha = 0.5  # learning rate
        self.gamma = 0.95  # favour future rewards
        self.rewards = {
            0: 1.0,  # No other agent finished before
            1: 0.6,  # One other agent finished before
            2: 0.5,  # Two other agents finished before
            3: -1.0,  # Three other agents finished before
        }

        # Q table
        self.qtable = QTable()

    def start_episode(self, initial_hand, num_episode=0):
        """ Initialize game with assigned initial hand. """

        super().start_episode(initial_hand, num_episode)

        # amount of random decisions
        self.epsilon = 1 / np.sqrt(num_episode / 2000 + 1)

    def save_model(self):
        """ Save the model to the specified path. """

        self.qtable.qtable.to_csv(
            "./egd/saved_agents/qtable_agent_"
            + str(self.playerIndex) + ".csv",
            header=None, index=False)

    def load_model(self):
        """ Load model from file. """

        self.qtable.restore_from_file(
            "./egd/saved_agents/qtable_agent_"
            + str(self.playerIndex) + ".csv")

    def decide_action_to_take(
            self, already_played, board, always_use_best,
            print_luck, possible_actions):
        """ Returns (possible_qvalues, action_index, action_taken, 
            random_choice, best_decision_made_randomly) """

        # Retrieve Q-Table for current state and add new if necessary
        learned_values = self.qtable.get_qtable_entry(
            already_played, board, self.hand)

        if self.debug:
            print("Player", self.playerIndex,
                  "- Learned Values", learned_values)

        # Do random decisions with a fixed probability
        best_decision_made_randomly = False
        if not always_use_best and np.random.uniform() < self.epsilon:
            random_decision = True
            action_index = np.random.choice(len(possible_actions))
        else:
            random_decision = False

            # Debug "luck"
            if (np.all(learned_values == None)
                    or np.all(learned_values.iloc[0, :] == 0)):
                best_decision_made_randomly = True
                if print_luck:
                    print("Player", self.playerIndex,
                          "- Warning: Decision made randomly")

            if np.any(learned_values != None):
                # Get best action with random tie-breaking
                possible_qvalues = learned_values.iloc[
                    0, list(range(len(possible_actions)))
                ]
                action_index = np.random.choice(
                    np.flatnonzero(np.isclose(
                        possible_qvalues, np.nanmax(possible_qvalues)
                    )))
            else:
                action_index = np.random.randint(len(possible_actions))

        action_taken = possible_actions[action_index]
        return (True, learned_values, action_index, action_taken,
                random_decision, best_decision_made_randomly)

    def process_next_board_state(
            # Last board state
            self, already_played, board,
            # Next board state
            next_already_played, next_board, next_hand,
            # Decided action
            learned_values, action_index, action_taken, random_choice,
            # Other parameters
            agents_finished, next_action_wins_board, always_use_best):
        """ Processes the next board state. """

        # Retrieve next state's q-value
        next_qvalues = self.qtable.get_qtable_entry(
            next_already_played, next_board, next_hand)
        next_max = np.nanmax(next_qvalues) \
            if np.any(next_qvalues != None) else 0

        # Determine reward
        if has_finished(next_hand):
            reward_earned = self.rewards[agents_finished]
        else:
            reward_earned = 0

        # Only update if either old or new values are not all zero
        if np.any(learned_values != None) or reward_earned != 0 or next_max != 0:
            # Create Q-Table entry if necessary
            if np.all(learned_values == None):
                self.qtable.create_qtable_entry(
                    already_played, board, self.hand)
                learned_values = self.qtable.get_qtable_entry(
                    already_played, board, self.hand)

            # Determine new value
            def update_func(old_qvalues):
                old_qvalue = old_qvalues.iloc[0, action_index]
                new_value = (1 - self.alpha) * old_qvalue + \
                    self.alpha * (reward_earned + self.gamma * next_max)
                old_qvalues.iloc[0, action_index] = new_value
                return old_qvalues

            self.qtable.update_qtable(
                already_played, board, self.hand, update_func)
