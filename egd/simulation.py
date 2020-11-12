import numpy as np
import pandas as pd
import argparse
import tqdm

from egd.game.cards import NUM_CARD_VALUES
from egd.game.state import NUM_PLAYERS, PLAYER, random_initial_cards
from egd.qlearning.agent import QLearningAgent


debug = False


def do_simulation(num_epochs):
    """ Simulates a given number of games. """

    agents = list(map(QLearningAgent, PLAYER))

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

        # Game loop
        while QLearningAgent.agents_finished < NUM_PLAYERS:
            # Perform a move
            finished, new_already_played, new_board = \
                agents[current_player_index].do_step(
                    already_played, board)

            if debug:
                print("Player", current_player_index, "- Board:", new_board)

            # Reset board after a complete round of no moves
            if np.all(new_already_played == already_played):
                turns_passed_without_move += 1

                if turns_passed_without_move == NUM_PLAYERS:
                    turns_passed_without_move = 0
                    new_board = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
            else:
                turns_passed_without_move = 0

            # Next turn
            current_player_index = (current_player_index + 1) % NUM_PLAYERS
            already_played = new_already_played
            board = new_board

        # Log progress
        if epoch % 100 == 0:
            print("Game Simulation", epoch, "out of", num_epochs)

    # Save trained agents
    for playerIndex, agent in enumerate(agents):
        agent.save_model(
            "./egd/saved-agents/qtable-agent-"
            + playerIndex + ".csv")


if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser(
        description="Hyperparameters for Training")
    parser.add_argument(
        'epochs', default=10000, type=int, nargs="?",
        metavar='Number of simulated games.')
    args = parser.parse_args()

    # Start simulation
    do_simulation(args.epochs)
