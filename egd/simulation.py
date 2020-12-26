import numpy as np
import pandas as pd
import tqdm
from typing import List, Tuple

from egd.config import epochs_for_validation, validation_games
from egd.io_util import log_initial_board, log_player_move, log_player_ranks
from egd.evaluation import print_validation_results
from egd.agent.state import GameState, StepOptions, EvaluationStats
from egd.agent.base_agent import ModelBase
from egd.game.state import random_initial_cards


def play_single_game(
        agents: List[ModelBase], epoch: int,
        verbose: bool, inference: bool) -> Tuple[List[int], np.ndarray]:
    """ Play a single round of the game.

    Args:
        agents (List[ModelBase]): Agents in the game.
        epoch (int): The current training epoch.
        verbose (bool): Verbosity flag.
        inference (bool): Whether to not train any agent.

    Returns:
        Tuple[List[int], np.ndarray]: Tuple of agent's 
        rankings and amount of random decisions
    """

    # Global game variables
    state = GameState()
    eval_stats = EvaluationStats()
    options = StepOptions(verbose, inference)

    # Generate and assign initial cards
    initial_cards = random_initial_cards()
    for playerIndex, agent in enumerate(agents):
        agent.start_episode(initial_cards[playerIndex], state, epoch)

    # Print intial board
    if verbose:
        log_initial_board(state)

    # Game loop
    while not state.all_agents_finished():
        # Perform a move
        decision_random = agents[state.current_player()].do_step(options)

        # Amount of random decisions for evaluation
        eval_stats.log_random_decision(state.current_player(), decision_random)

        # Print move
        if verbose:
            log_player_move(state)

        # Reset board after a complete round of no moves
        state.check_all_passed(verbose)

        # Next turn
        state.next_turn()

    # Game finished
    if verbose:
        log_player_ranks(state)

    # Return ranking of game
    return state.agent_ranking, eval_stats.get_amount_of_random_decisions()


# TODO refactor
def do_simulation(
        agents: List[ModelBase], agent_strings: List[str], num_epochs: int,
        verbose: bool, save_model: bool, inference: bool):
    """ Simulates a given number of games. """

    # Stats containing epoch, agent index, agent name, mean rank,
    # mean amount of random decisions, quality of q-value approximation
    simulation_stats = []

    for epoch in tqdm.tqdm(range(num_epochs)):
        # Play single game
        play_single_game(agents, epoch, verbose, inference)

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
