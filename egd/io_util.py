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
