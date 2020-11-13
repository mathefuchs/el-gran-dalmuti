from egd.simple_agents.human_agent import HumanAgent
from egd.simple_agents.simple_agent import SimpleAgent
from egd.qlearning.agent import QLearningAgent


def get_agent(player_index, agent_string, load_model):
    """ Get agent for string identifier. """

    if agent_string == "Human":
        agent = HumanAgent(player_index)
    elif agent_string == "Simple":
        agent = SimpleAgent(player_index)
    elif agent_string == "QLearningAgent":
        agent = QLearningAgent(player_index)
    elif agent_string == "DeepQAgent":
        raise NotImplementedError("Deep Q Agent has not been implemented yet.")
    else:
        raise Exception("Agent does not exist:", agent_string)

    if load_model:
        agent.load_model()

    return agent
