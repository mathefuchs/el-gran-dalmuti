from egd.agent.state import GameState
from egd.agent.base_agent import ModelBase
from egd.agent.simple_agents.human_agent import HumanAgent
from egd.agent.simple_agents.simple_agent import SimpleAgent
from egd.agent.simple_agents.random_agent import RandomAgent
from egd.agent.deep_qlearning.deepq_agent import DeepQAgent


def get_agent(
        player_index: int, agent_string: str,
        load_model: bool) -> ModelBase:
    """ Get agent for string identifier.

    Args:
        player_index (int): The player's index.
        agent_string (str): The string 
        representing the type of agent.
        load_model (bool): Whether to 
        load a previously trained model.

    Raises:
        Exception: If the agent with 
        the given name does not exist.

    Returns:
        ModelBase: The agent.
    """

    if agent_string == "Human":
        agent = HumanAgent(player_index)
    elif agent_string == "Simple":
        agent = SimpleAgent(player_index)
    elif agent_string == "Random":
        agent = RandomAgent(player_index)
    elif agent_string == "DeepQAgent":
        agent = DeepQAgent(player_index, create_model=(not load_model))
    else:
        raise Exception("Agent does not exist:", agent_string)

    if load_model:
        agent.load_model()

    return agent


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
