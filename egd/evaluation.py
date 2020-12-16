import numpy as np

from egd.game.state import NUM_PLAYERS


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
