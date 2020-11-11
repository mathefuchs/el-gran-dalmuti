import numpy as np
import pandas as pd


single_state_width = 13
state_columns = 3 * single_state_width
next_states = 31
num_columns = state_columns + next_states

already_played_indices = list(range(0, single_state_width))
board_indices = list(range(single_state_width, 2 * single_state_width))
hand_indices = list(range(2 * single_state_width, 3 * single_state_width))
next_state_indices = list(range(state_columns, state_columns + next_states))


def query_qtable(qtable, already_played, board, hand):
    """ Queries the q-table. """

    return np.all(
        qtable.iloc[:, already_played_indices] == already_played
        & qtable.iloc[:, board_indices] == board
        & qtable.iloc[:, hand_indices] == hand, axis=1
    )


def create_qtable_df():
    """ Creates an empty Q-table. """

    # Create empty dataframe
    qtable = pd.DataFrame(columns=list(range(num_columns)))

    # Set types
    for column in range(state_columns):
        qtable[column] = qtable[column].astype(np.int8)
    for column in range(next_states):
        qtable[state_columns + column] = qtable[column].astype(np.float32)

    return qtable


def create_qtable_entry(qtable, already_played, board, hand):
    """ Creates a new qtable entry. """

    # Create new entry row
    new_qtable_entry = pd.DataFrame(
        np.zeros((1, next_states)), index=[0],
        columns=list(range(num_columns)))

    # Set data types
    for column in range(state_columns):
        new_qtable_entry[column] = new_qtable_entry[column].astype(np.int8)
    for column in range(next_states):
        new_qtable_entry[state_columns + column] = \
            new_qtable_entry[column].astype(np.float32)

    return qtable.append(new_qtable_entry, ignore_index=True)


def get_qtable_entry(qtable, already_played, board, hand):
    """ Get q-values for entry. """

    result = qtable[query_qtable(qtable, already_played, board, hand)]

    if len(result) == 1:
        return result[next_state_indices]
    else:
        return None


def update_qtable(qtable, already_played, board, hand, update_qvalue_func):
    """ Update q-value. """

    row_selected = query_qtable(qtable, already_played, board, hand)
    qtable.iloc[row_selected, next_state_indices] = update_qvalue_func(
        qtable.iloc[row_selected, next_state_indices]
    )


def delete_qtable_entry(qtable, already_played, board, hand):
    """ Delete q-table entry. """

    qtable.drop(qtable.index[
        query_qtable(qtable, already_played, board, hand)
    ], inplace=True)
