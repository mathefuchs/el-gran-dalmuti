import numpy as np
import pandas as pd
import argparse
import tqdm

from egd.game.cards import NUM_CARD_VALUES
from egd.game.state import NUM_PLAYERS, PLAYER, random_initial_cards
from egd.game.moves import possible_next_moves, only_passing_possible
from egd.util import get_agent


def play_single_game(agents, epoch, verbose, inference):
    """ Play a single round of the game. """

    # Generate and assign initial cards
    initial_cards = random_initial_cards()
    for playerIndex, agent in enumerate(agents):
        agent.start_episode(initial_cards[playerIndex], epoch)

    # Random first player and order of play
    order_of_play = np.random.permutation(PLAYER)
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
    number_decisions = np.zeros(NUM_PLAYERS, dtype=np.int16)
    best_decisions_randomly = np.zeros(NUM_PLAYERS, dtype=np.int16)
    while len(finished_players) < NUM_PLAYERS:
        # Current player
        current_player = order_of_play[current_player_index]
        next_players = [order_of_play[
            (current_player_index + i) %
            NUM_PLAYERS] for i in range(1, NUM_PLAYERS)]

        # Check whether action wins board
        def next_action_wins_board(next_already_played, next_board):
            next_player_index = 0
            while next_player_index < NUM_PLAYERS - 1 and only_passing_possible(
                    agents[next_players[next_player_index]].hand, next_board):
                next_player_index += 1

            if next_player_index == NUM_PLAYERS - 1:
                return True
            else:
                return False

        # Perform a move
        finished, new_already_played, new_board, best_dec_rand = \
            agents[current_player].do_step(
                already_played, board, len(finished_players),
                next_action_wins_board=next_action_wins_board,
                always_use_best=inference, print_luck=verbose)

        # Amount of random decisions for evaluation
        number_decisions[current_player] += 1
        best_decisions_randomly[current_player] += 1 \
            if best_dec_rand else 0

        # Keep track of finished agents
        if finished and current_player not in finished_players:
            finished_players.append(current_player)

        # Print move
        if verbose:
            if finished:
                print("Player", current_player,
                      "- Board:", new_board, "- Finished")
            elif np.all(new_already_played == already_played):
                print("Player", current_player,
                      "- Board:", new_board, "- Passed")
            else:
                print("Player", current_player,
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

    # Return ranking of game
    return finished_players, best_decisions_randomly / number_decisions


def do_simulation(agents, agent_strings, num_epochs,
                  verbose, save_model, inference):
    """ Simulates a given number of games. """

    # Stats containing epoch, agent index, agent name, mean rank,
    # mean amount of random decisions, quality of q-value approximation
    simulation_stats = []

    for epoch in tqdm.tqdm(range(num_epochs)):
        # Play single game
        play_single_game(agents, epoch, verbose, inference)

        # Validate progress each 1000 games
        if epoch != 0 and epoch % 10 == 0:  # 1000
            # Play 1000 games with best decisions
            if verbose:
                print()
                print("Validation - Epoch ", epoch, ":", sep="")

            # Log general stats
            agent_stats = []
            for i, agent in enumerate(agents):
                agent_stats.append([epoch, i, agent_strings[i]])

            # Evaluate rankings
            rankings = []
            rand_amounts = []
            for _ in tqdm.tqdm(range(2)):  # 100
                ranking, rand_amount = play_single_game(
                    agents, 0, False, True)
                rankings.append(ranking)
                rand_amounts.append(rand_amount)
            print_validation_results(
                epoch, rankings, rand_amounts,
                agent_strings, agent_stats, verbose)

            # Evaluate agent's q-value representation
            for i, agent in enumerate(agents):
                metrics = agent.evaluate_inference_mode()
                agent_stats[i].extend(metrics[1:] if metrics else [0.0, 0.0])

            # Append test stats
            simulation_stats.extend(agent_stats)

            # Save every 1000 epochs
            if save_model:
                for agent in agents:
                    agent.save_model()

    if save_model:
        # Save training stats
        stats_df = pd.DataFrame(
            simulation_stats, columns=[
                "epoch", "agent", "agent_name",
                "mean_rank", "mean_rand_decisions",
                "huber_loss_q_val_approx", "mse_loss_q_val_approx"
            ])
        stats_df.to_csv(
            "./egd/saved_agents/training_deepq_stats.csv",
            index=False)

        # Save trained agents
        for agent in agents:
            agent.save_model()


def print_validation_results(
        epoch, rankings, rand_amounts,
        agent_strings, agent_stats, verbose):
    """ Prints validation results for the given agents. """

    # Compute mean ranks and amount of random decisions
    mean_ranks = [
        np.mean(np.where(np.array(rankings) == player_index)[1])
        for player_index in range(NUM_PLAYERS)
    ]
    mean_rand_dec = np.mean(np.vstack(rand_amounts), axis=0)

    # Populate stats
    for i, entry in enumerate(agent_stats):
        entry.extend([mean_ranks[i], mean_rand_dec[i]])

    # Zip names and metrics for readibility
    rank_and_name = list(zip(agent_strings, mean_ranks))
    mean_rand_name = list(zip(agent_strings, mean_rand_dec))

    if verbose:
        print()
        print(epoch)
        print("Player's mean ranks", rank_and_name)
        print("Player's amount of random decisions", mean_rand_name)
    else:
        print()
        print(epoch)
        print(rank_and_name)
        print(mean_rand_name)


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
    do_simulation(
        agents, agent_strings, args.games, (args.verbose == 1),
        (args.savemodel == 1), (args.inference == 1))
