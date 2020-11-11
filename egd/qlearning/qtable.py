import numpy as np
import pandas as pd


class QTable:
    single_state_width = 13
    state_columns = 3 * single_state_width
    # 236 = (1 + 2 + ... + 12) * 3 [Combination with Jokers] + 2 [Jokers played alone]
    next_states = 236
    num_columns = state_columns + next_states

    already_played_indices = list(range(0, single_state_width))
    board_indices = list(range(single_state_width, 2 * single_state_width))
    hand_indices = list(range(2 * single_state_width, 3 * single_state_width))
    next_state_indices = list(
        range(state_columns, state_columns + next_states))

    def __init__(self):
        """ Creates an empty Q-table. """

        # Create empty dataframe
        self.qtable = pd.DataFrame(columns=list(range(QTable.num_columns)))

        # Set types
        for column in range(QTable.state_columns):
            self.qtable[column] = self.qtable[column].astype(np.int8)
        for column in range(QTable.next_states):
            self.qtable[QTable.state_columns + column] = \
                self.qtable[column].astype(np.float32)

    def _query_qtable(self, already_played, board, hand):
        """ Queries the q-table. """

        return np.all(
            (self.qtable.iloc[:, QTable.already_played_indices]
             == already_played).values
            & (self.qtable.iloc[:, QTable.board_indices] == board).values
            & (self.qtable.iloc[:, QTable.hand_indices] == hand).values, axis=1
        )

    def create_qtable_entry(self, already_played, board, hand):
        """ Creates a new qtable entry. """

        # Create new entry row
        new_qtable_entry = pd.DataFrame(
            np.hstack([
                already_played, board,
                hand, np.zeros(QTable.next_states)
            ]).reshape((1, QTable.num_columns)),
            index=[0], columns=list(range(QTable.num_columns)))

        # Set data types
        for column in range(QTable.state_columns):
            new_qtable_entry[column] = new_qtable_entry[column].astype(np.int8)
        for column in range(QTable.next_states):
            new_qtable_entry[QTable.state_columns + column] = \
                new_qtable_entry[QTable.state_columns +
                                 column].astype(np.float32)

        self.qtable = self.qtable.append(new_qtable_entry, ignore_index=True)

    def get_qtable_entry(self, already_played, board, hand):
        """ Get q-values for entry. """

        if len(self.qtable) == 0:
            return None

        result = self.qtable.iloc[self._query_qtable(
            already_played, board, hand), QTable.next_state_indices]

        if len(result) == 1:
            return result
        else:
            return None

    def update_qtable(self, already_played, board, hand, update_qvalue_func):
        """ Update q-value. """

        if len(self.qtable) == 0:
            return

        row_selected = self._query_qtable(already_played, board, hand)
        self.qtable.iloc[row_selected, QTable.next_state_indices] = \
            update_qvalue_func(
                self.qtable.iloc[row_selected, QTable.next_state_indices])

    def delete_qtable_entry(self, already_played, board, hand):
        """ Delete q-table entry. """

        if len(self.qtable) == 0:
            return

        self.qtable.drop(self.qtable.index[
            self._query_qtable(already_played, board, hand)
        ], inplace=True)
