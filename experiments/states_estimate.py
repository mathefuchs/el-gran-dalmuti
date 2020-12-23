import numpy as np
import tqdm

from egd.game.state import random_initial_cards
from egd.game.cards import NUM_CARD_VALUES, AVAILABLE_CARDS
from egd.game.moves import possible_next_moves


def num_state_action_comb():
    """ Estimate how many state-action combinations there are 
        to evaluate the feasibility of a table-based q-learning approach. """

    empty = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
    total_comb = []

    comb_ap = 1
    for i in range(NUM_CARD_VALUES):
        comb_ap *= (AVAILABLE_CARDS[i] + 1)

    for _ in tqdm.tqdm(range(10000)):
        # Already played can be anything
        ap = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
        for i in range(NUM_CARD_VALUES):
            ap[i] = np.random.randint(1, AVAILABLE_CARDS[i] + 1)

        # Board must be a move that is already included in already played
        comb_b = possible_next_moves(ap, empty).shape[0]

        # Actions from other agents must be within the remaining cards
        remaining = AVAILABLE_CARDS - ap
        hands = random_initial_cards(cards_to_use=remaining)
        comb_actions = 1
        for hand in hands:
            comb_actions *= possible_next_moves(hand, empty).shape[0]

        total_comb.append(comb_ap * comb_b * comb_actions)

    # 68,319,447,356,160,000
    print(max(total_comb))
    #    138,538,758,758,400
    print(min(total_comb))


if __name__ == "__main__":
    # Estimate size of state and action space
    num_state_action_comb()
