import numpy as np


NUM_CARD_VALUES = 13
JOKER = 12  # Jokers at index 12
AVAILABLE_CARDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2])


def get_cards_array(card_type, num_cards):
    """ Vector representation of the cards of one kind. """

    cards_array = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
    cards_array[card_type] = num_cards
    return cards_array
