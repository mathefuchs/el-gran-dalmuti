import numpy as np
import pandas as pd
import argparse
import tqdm

from egd.game.cards import NUM_CARD_VALUES
from egd.game.state import NUM_PLAYERS, random_initial_cards
from egd.game.moves import possible_next_moves, only_passing_possible
from egd.util import get_agent


def do_simulation(agents, num_epochs, verbose, save_model, inference):
    """ Simulates a given number of games. """

    for epoch in tqdm.tqdm(range(num_epochs)):
        # Generate and assign initial cards
        initial_cards = random_initial_cards()
        for playerIndex, agent in enumerate(agents):
            agent.start_episode(initial_cards[playerIndex], epoch)

        # Random first player
        current_player_index = np.random.randint(NUM_PLAYERS)
        already_played = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
        board = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
        turns_passed_without_move = 0

        # Print intial board
        if verbose:
            print("                    1 2 3 4 5 6 7 8 9 . . . J")
            print("Initial board:   ", board)

        # Game loop
        finished_players = []
        while len(finished_players) < NUM_PLAYERS:
            # Check whether action wins board
            def next_action_wins_board(next_already_played, next_board):
                next_players = [(current_player_index + i) %
                                NUM_PLAYERS for i in range(1, NUM_PLAYERS)]
                next_player_index = 0

                while next_player_index < NUM_PLAYERS - 1 and only_passing_possible(
                        agents[next_players[next_player_index]].hand, next_board):
                    next_player_index += 1

                if next_player_index == NUM_PLAYERS - 1:
                    return True
                else:
                    return False

            # Perform a move
            finished, new_already_played, new_board = \
                agents[current_player_index].do_step(
                    already_played, board, len(finished_players),
                    next_action_wins_board=next_action_wins_board,
                    always_use_best=inference, print_luck=verbose)

            # Keep track of finished agents
            if finished and current_player_index not in finished_players:
                finished_players.append(current_player_index)

            # Print move
            if verbose:
                if finished:
                    print("Player", current_player_index,
                          "- Board:", new_board, "- Finished")
                elif np.all(new_already_played == already_played):
                    print("Player", current_player_index,
                          "- Board:", new_board, "- Passed")
                else:
                    print("Player", current_player_index,
                          "- Board:", new_board)

            # Reset board after a complete round of no moves
            if np.all(new_already_played == already_played):
                turns_passed_without_move += 1

                if turns_passed_without_move == NUM_PLAYERS - 1:
                    turns_passed_without_move = 0
                    new_board = np.zeros(NUM_CARD_VALUES, dtype=np.int8)

                    if verbose:
                        print("                   1 2 3 4 5 6 7 8 9 . . . J")
                        print("New board:       ", new_board)
            else:
                turns_passed_without_move = 0

            # Next turn
            current_player_index = (current_player_index + 1) % NUM_PLAYERS
            already_played = new_already_played
            board = new_board

        # Game finished
        if verbose:
            print("Game finished - Player's Ranks", finished_players)

        # Save every 100 epochs
        if save_model and epoch % 100 == 0:
            for agent in agents:
                agent.save_model()

    # Save trained agents
    if save_model:
        for agent in agents:
            agent.save_model()


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
    do_simulation(agents, args.games, (args.verbose == 1),
                  (args.savemodel == 1), (args.inference == 1))
