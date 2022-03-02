from typing import List, Tuple

import numpy as np
import pandas as pd
import tqdm

from egd.agent.base_agent import ModelBase
from egd.config import epochs_for_validation, validation_games
from egd.evaluation import print_validation_results
from egd.game.cards import NUM_CARD_VALUES
from egd.game.state import NUM_PLAYERS, PLAYER, random_initial_cards
from egd.game.moves import possible_next_moves_for_all
from egd.simulation_extract import GameExtractor


def play_single_game(
    agents: List[ModelBase], epoch: int, verbose: bool, inference: bool,
    game_records: GameExtractor, save_histories: bool
) -> Tuple[List[int], float]:
    """ Play a single round of the game. """

    # Generate and assign initial cards
    initial_cards = random_initial_cards()
    for playerIndex, agent in enumerate(agents):
        agent.start_episode(initial_cards[playerIndex], epoch)

    # Random first player and order of play
    order_of_play = np.random.permutation(PLAYER)
    current_player_index = np.random.randint(NUM_PLAYERS)
    current_player = order_of_play[current_player_index]
    already_played = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
    board = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
    turns_passed_without_move = 0

    # Print initial board
    if verbose:
        print("                    1 2 3 4 5 6 7 8 9 . . . J")
        print("Initial board:   ", board)

    # Global game variables
    past_actions = np.zeros(
        (NUM_PLAYERS, NUM_CARD_VALUES), dtype=np.int8)  # TODO
    finished_players = []

    # Statistics to evaluate
    number_decisions = np.zeros(NUM_PLAYERS, dtype=np.int16)
    best_decisions_randomly = np.zeros(NUM_PLAYERS, dtype=np.int16)

    # Game loop
    while len(finished_players) < NUM_PLAYERS:
        # Current player
        current_player = order_of_play[current_player_index]
        next_players = [order_of_play[
            (current_player_index + i) %
            NUM_PLAYERS] for i in range(1, NUM_PLAYERS)]

        # List possible states before the next action of the same agent
        def states_before_next_action(ap, b):
            next_aps = [ap]
            next_bs = [b]

            # FIXME data leakage, hand of other players should be secret
            for next_player in next_players:
                _, next_bs, next_aps = possible_next_moves_for_all(
                    agents[next_player].hand, next_bs, next_aps)

            return next_aps, next_bs

        # Perform a move
        finished, new_already_played, new_board, best_dec_rand = \
            agents[current_player].do_step(
                already_played, board, len(finished_players),
                list_next_possible_states=states_before_next_action,
                always_use_best=inference, print_luck=verbose)

        # Amount of random decisions for evaluation
        number_decisions[current_player] += 1
        best_decisions_randomly[current_player] += 1 \
            if best_dec_rand else 0

        # Record first game state
        if save_histories and current_player not in finished_players:
            game_records.record_state(
                epoch, current_player, new_already_played,
                new_board, agents[current_player].hand
            )

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

    # Record game ranking
    if save_histories:
        game_records.set_game_ranking(finished_players)
        if (epoch + 1) % 10000 == 0:
            game_records.export_log_to_file(epoch)

    # Return ranking of game
    return finished_players, best_decisions_randomly / number_decisions


def do_simulation(
    agents: List[ModelBase], agent_strings: List[str], num_epochs: int,
    verbose: bool, save_model: bool, inference: bool, save_histories: bool
):
    """ Simulates a given number of games. """

    # Stats containing epoch, agent index, agent name, mean rank,
    # mean amount of random decisions, quality of q-value approximation
    simulation_stats = []
    game_records = GameExtractor()

    for epoch in tqdm.tqdm(range(num_epochs)):
        # Play single game
        play_single_game(agents, epoch, verbose, inference,
                         game_records, save_histories)

        # Validate progress each x games
        if epoch != 0 and epoch % epochs_for_validation == 0:
            # Play x games with best decisions
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
            for _ in tqdm.tqdm(range(validation_games)):
                ranking, rand_amount = play_single_game(
                    agents, 0, False, True, None, False)
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
