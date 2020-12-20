import numpy as np

from egd.game.cards import NUM_CARD_VALUES, AVAILABLE_CARDS, get_cards_array


NUM_PLAYERS = 4
PLAYER = list(range(NUM_PLAYERS))


def has_finished(hand):
    """ 
        Whether the hand is already empty.

        hand - vector with 13 entries (number of 1, 2, ..., 12, Jokers)
    """

    if len(hand.shape) == 1:
        return np.all(hand == 0)
    else:
        return np.all(hand == 0, axis=1)


def random_initial_cards():
    """ Random initial state for the game. """

    deck = np.array([], dtype=np.int8)

    for card_type in range(NUM_CARD_VALUES):
        deck = np.append(deck, np.array(
            [card_type for _ in range(AVAILABLE_CARDS[card_type])]))

    np.random.shuffle(deck)

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

    return player_initial_hands
