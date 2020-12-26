from egd.agent.state import GameState


def log_initial_board(state: GameState):
    """ Logs the initial board state.

    Args:
        state (GameState): The game state.
    """

    print("                    1 2 3 4 5 6 7 8 9 . . . J")
    print("Initial board:   ", state.curr_board)


def log_player_move(state: GameState):
    """ Logs the last player's move.

    Args:
        state (GameState): The game state.
    """

    if state.agents_finished[state.current_player()]:
        print("Player", state.current_player(),
              "- Board:", state.curr_board, "- Finished")
    elif state.last_action_passing():
        print("Player", state.current_player(),
              "- Board:", state.curr_board, "- Passed")
    else:
        print("Player", state.current_player(),
              "- Board:", state.curr_board)


def log_board_reset(state: GameState):
    """ Logs the board after a reset.

    Args:
        state (GameState): The game state.
    """

    print("                   1 2 3 4 5 6 7 8 9 . . . J")
    print("New board:       ", state.curr_board)


def log_player_ranks(state: GameState):
    """ Logs the player's rankings.

    Args:
        state (GameState): The game state.
    """

    print("Game finished - Player's Ranks", state.agent_ranking)
