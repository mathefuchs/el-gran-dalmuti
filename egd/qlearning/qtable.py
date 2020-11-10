import numpy as np
import pandas as pd


single_state_width = 13
state_columns = 3 * single_state_width
next_states = 31
num_columns = state_columns + next_states

already_played_indices = list(range(0, single_state_width))
board_indices = list(range(single_state_width, 2 * single_state_width))
hand_indices = list(range(2 * single_state_width, 3 * single_state_width))


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


def add_or_update_qtable(qtable, already_played, board, hand, update_qvalue_func):
    """ If entry already exists, update q-value. Else, create new entry. """

    pass


def get_qtable_entry(qtable, already_played, board, hand):
    """ Get q-values for entry. """

    pass
