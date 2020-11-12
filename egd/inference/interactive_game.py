import numpy as np
import pandas as pd
import argparse
import tqdm

from egd.game.cards import NUM_CARD_VALUES
from egd.game.state import NUM_PLAYERS, PLAYER, random_initial_cards
from egd.qlearning.agent import QLearningAgent
from egd.inference.human_agent import HumanAgent


def get_agent(player_index, agent_string):
    """ Get agent for string identifier. """

    if agent_string == "Human":
        return HumanAgent(player_index)
    elif agent_string == "QLearningAgent":
        agent = QLearningAgent(player_index)
        agent.load_model("./egd/saved-agents/qtable-agent-" +
                         str(player_index) + ".csv")
        return agent
    elif agent_string == "DeepQAgent":
        raise NotImplementedError("Deep Q Agent has not been implemented yet.")
    else:
        raise Exception("Agent does not exist:", agent_string)


def do_simulation(agents):
    """ Simulates a single game with the given agents. """

    # Generate and assign initial cards
    initial_cards = random_initial_cards()
    for playerIndex, agent in enumerate(agents):
        agent.start_episode(initial_cards[playerIndex])

    # Random first player
    current_player_index = np.random.randint(NUM_PLAYERS)
    already_played = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
    board = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
    turns_passed_without_move = 0

    # Print intial board
    print("                   1 2 3 4 5 6 7 8 9 . . . J")
    print("Initial board:   ", board)

    # Game loop
    finished_players = []
    while len(finished_players) < 4:
        # Perform a move
        finished, new_already_played, new_board = \
            agents[current_player_index].do_step(
                already_played, board, always_use_best=True,
                print_luck=True)

        # Keep track of finished agents
        if finished and current_player_index not in finished_players:
            finished_players.append(current_player_index)

        # Print move
        if finished:
            print("Player", current_player_index,
                  "- Board:", new_board, "- Finished")
        elif np.all(new_already_played == already_played):
            print("Player", current_player_index,
                  "- Board:", new_board, "- Passed")
        else:
            print("Player", current_player_index, "- Board:", new_board)

        # Reset board after a complete round of no moves
        if np.all(new_already_played == already_played):
            turns_passed_without_move += 1

            if turns_passed_without_move == NUM_PLAYERS - 1:
                turns_passed_without_move = 0
                new_board = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
                print("                   1 2 3 4 5 6 7 8 9 . . . J")
                print("New board:       ", new_board)
        else:
            turns_passed_without_move = 0

        # Next turn
        current_player_index = (current_player_index + 1) % NUM_PLAYERS
        already_played = new_already_played
        board = new_board

    # Game finished
    print("Game finished - Player's Ranks", finished_players)


if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser(
        description="Selection of Agents")
    parser.add_argument(
        'player1', default="Human", type=str, nargs="?",
        metavar='Type of player 1.')
    parser.add_argument(
        'player2', default="QLearningAgent", type=str, nargs="?",
        metavar='Type of player 2.')
    parser.add_argument(
        'player3', default="QLearningAgent", type=str, nargs="?",
        metavar='Type of player 3.')
    parser.add_argument(
        'player4', default="QLearningAgent", type=str, nargs="?",
        metavar='Type of player 4.')
    args = parser.parse_args()

    # Parse agents
    agent_strings = [args.player1, args.player2,
                     args.player3, args.player4]
    agents = []
    for player_index in range(NUM_PLAYERS):
        agents.append(get_agent(
            player_index, agent_strings[player_index]))

    # Start simulation
    do_simulation(agents)
