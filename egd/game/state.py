import numpy as np

from egd.game.cards import NUM_CARD_VALUES, AVAILABLE_CARDS, get_cards_array


NUM_PLAYERS = 4
PLAYERS = list(range(NUM_PLAYERS))


def has_finished(hand: np.ndarray) -> bool:
    """ Whether the hand is already empty.

    Args:
        hand (np.ndarray): Vector with 13 entries (count of 1, 2, ..., 12, Jokers)

    Returns:
        bool: Whether the hand is already empty
    """

    if len(hand.shape) == 1:
        return np.all(hand == 0)
    else:
        return np.all(hand == 0, axis=1)


def random_initial_cards(cards_to_use=AVAILABLE_CARDS) -> np.ndarray:
    """ Random initial state for the game.

    Args:
        cards_to_use (np.ndarray, optional): Card stack to use. 
        Defaults to AVAILABLE_CARDS.

    Returns:
        np.ndarray: NUM_PLAYERS rows with the shuffled cards
    """

    # Insert each individual card into the deck
    deck = np.array([], dtype=np.int8)
    for card_type in range(NUM_CARD_VALUES):
        deck = np.append(deck, np.array(
            [card_type for _ in range(cards_to_use[card_type])],
            dtype=np.int8))

    # Shuffle deck
    np.random.shuffle(deck)

    # Divide the card deck
    chunk = deck.shape[0] // NUM_PLAYERS
    remainder = deck.shape[0] % NUM_PLAYERS
    first_player_initialized = False

    for playerIndex in range(NUM_PLAYERS):
        beginOfChunk = playerIndex * chunk + min(playerIndex, remainder)
        endOfChunk = (playerIndex + 1) * chunk + \
            min(playerIndex + 1, remainder)
        player = np.zeros(NUM_CARD_VALUES, dtype=np.int8)

        for card in deck[beginOfChunk:endOfChunk]:
            player += get_cards_array(card, 1)

        if first_player_initialized:
            player_initial_hands = np.vstack([player_initial_hands, player])
        else:
            first_player_initialized = True
            player_initial_hands = player

    # Return NUM_PLAYERS rows with the shuffled cards
    return player_initial_hands
