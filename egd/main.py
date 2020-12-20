import argparse

from egd.io_util import get_agent
from egd.game.state import NUM_PLAYERS
from egd.simulation import do_simulation
from egd.parallel_simulation import do_par_simulation


if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser(
        description="Selection of Agents")
    parser.add_argument(
        '--player0', default="Human", type=str, nargs="?",
        metavar='Type of player 0.')
    parser.add_argument(
        '--player1', default="QLearningAgent", type=str, nargs="?",
        metavar='Type of player 1.')
    parser.add_argument(
        '--player2', default="QLearningAgent", type=str, nargs="?",
        metavar='Type of player 2.')
    parser.add_argument(
        '--player3', default="QLearningAgent", type=str, nargs="?",
        metavar='Type of player 3.')
    parser.add_argument(
        '--games', default="100", type=int, nargs="?",
        metavar='Number of games.')
    parser.add_argument(
        '--verbose', default=0, type=int, nargs="?",
        metavar='Use verbose logging.')
    parser.add_argument(
        '--loadmodel', default=0, type=int, nargs="?",
        metavar='Whether to load trained models.')
    parser.add_argument(
        '--savemodel', default=0, type=int, nargs="?",
        metavar='Whether to save models.')
    parser.add_argument(
        '--inference', default=0, type=int, nargs="?",
        metavar='Whether to use the agents in inference mode.')
    parser.add_argument(
        '--parallel', default=0, type=int, nargs="?",
        metavar='Whether to use parallel processing.')
    args = parser.parse_args()

    # Parse agents
    agent_strings = [args.player0, args.player1,
                     args.player2, args.player3]
    agents = []
    for player_index in range(NUM_PLAYERS):
        agents.append(get_agent(
            player_index, agent_strings[player_index],
            (args.loadmodel == 1)))

    # Start simulation
    sim_call = do_simulation if args.parallel == 0 else do_par_simulation
    sim_call(agents, agent_strings, args.games, (args.verbose == 1),
             (args.savemodel == 1), (args.inference == 1))
