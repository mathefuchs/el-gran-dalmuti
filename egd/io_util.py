from egd.agent.simple_agents.human_agent import HumanAgent
from egd.agent.simple_agents.simple_agent import SimpleAgent
from egd.agent.simple_agents.random_agent import RandomAgent
from egd.agent.qlearning.qagent import QLearningAgent
from egd.agent.deep_qlearning.deepq_agent import DeepQAgent


def get_agent(player_index, agent_string, load_model):
    """ Get agent for string identifier. """

    if agent_string == "Human":
        agent = HumanAgent(player_index)
    elif agent_string == "Simple":
        agent = SimpleAgent(player_index)
    elif agent_string == "Random":
        agent = RandomAgent(player_index)
    elif agent_string == "QLearningAgent":
        agent = QLearningAgent(player_index)
    elif agent_string == "DeepQAgent":
        agent = DeepQAgent(player_index, create_model=(not load_model))
    else:
        raise Exception("Agent does not exist:", agent_string)

    if load_model:
        agent.load_model()

    return agent
