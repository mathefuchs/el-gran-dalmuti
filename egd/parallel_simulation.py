import numpy as np
import pandas as pd
import tqdm

from egd.config import epochs_for_validation, validation_games
from egd.evaluation import print_validation_results
from egd.game.cards import NUM_CARD_VALUES
from egd.game.state import NUM_PLAYERS, random_initial_cards
from egd.game.moves import possible_next_moves


def do_par_simulation(agents, agent_strings, num_epochs,
                      verbose, save_model, inference):
    """ Simulates a given number of games. """

    raise NotImplementedError()
