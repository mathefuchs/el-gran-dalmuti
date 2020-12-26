import numpy as np


NUM_CARD_VALUES = 13
JOKER = 12  # Jokers at index 12
AVAILABLE_CARDS = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2], dtype=np.int8)
EMPTY_HAND = np.zeros(NUM_CARD_VALUES, dtype=np.int8)


def get_cards_array(card_type: int, num_cards: int) -> np.ndarray:
    """ Vector representation of the cards of one kind.

    Args:
        card_type (int): The kind of cards
        num_cards (int): The number of cards of those kind

    Returns:
        np.ndarray: A vector representation of the given information
    """

    cards_array = np.zeros(NUM_CARD_VALUES, dtype=np.int8)
    cards_array[card_type] = num_cards
    return cards_array
