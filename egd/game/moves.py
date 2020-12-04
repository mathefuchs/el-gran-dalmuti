import numpy as np

from egd.game.cards import NUM_CARD_VALUES, JOKER
from egd.game.state import has_finished, get_cards_array


def only_passing_possible(hand, board):
    """
        Faster than checking len(possible_next_moves(...)) == 1.
    """

    # Finished players have to pass
    if has_finished(hand):
        return True

    # No passing allowed when board empty
    if np.all(board == 0):
        return False

    # Else iterate possible actions
    card_type_in_board = np.argmax(board)
    num_cards_in_board = board[card_type_in_board] \
        if card_type_in_board == JOKER \
        else board[card_type_in_board] + board[JOKER]

    for card_type_in_hand in range(NUM_CARD_VALUES - 1, -1, -1):
        # You can play clean
        if card_type_in_hand < card_type_in_board and \
                hand[card_type_in_hand] >= num_cards_in_board:
            return False

        # Or you can play dirty (with Joker(s))
        if card_type_in_hand != JOKER and hand[JOKER] > 0 \
                and card_type_in_hand < card_type_in_board \
                and num_cards_in_board >= 2 and hand[card_type_in_hand] > 0 \
                and hand[card_type_in_hand] + hand[JOKER] >= num_cards_in_board:
            # Use one joker
            if hand[card_type_in_hand] + 1 >= num_cards_in_board:
                return False

            # Use two jokers
            if hand[JOKER] == 2 and num_cards_in_board >= 3:
                return False

    # No possible actions available
    return True


def possible_next_moves(hand, board):
    """
        Returns possible next moves as a list of actions
    """

    # You can always pass if it is not the initial move
    possible_actions = [np.zeros((1, NUM_CARD_VALUES), dtype=np.int8)]

    # If board empty, moves do only depend on hand
    if np.all(board == 0):
        for card_type in range(NUM_CARD_VALUES - 1, -1, -1):
            for num_cards in range(hand[card_type], 0, -1):
                if card_type != JOKER:
                    for num_jokers in range(hand[JOKER] + 1):
                        # Form new board out of jokers and cards
                        possible_actions.append(
                            get_cards_array(card_type, num_cards)
                            + get_cards_array(JOKER, num_jokers)
                        )
                else:
                    # Form new board out of only jokers
                    possible_actions.append(
                        get_cards_array(card_type, num_cards)
                    )

    # Move has to match current board
    else:
        card_type_in_board = np.argmax(board)
        num_cards_in_board = board[card_type_in_board] \
            if card_type_in_board == JOKER \
            else board[card_type_in_board] + board[JOKER]

        if not has_finished(hand):
            for card_type_in_hand in range(NUM_CARD_VALUES - 1, -1, -1):
                # You can play clean
                if card_type_in_hand < card_type_in_board and \
                        hand[card_type_in_hand] >= num_cards_in_board:
                    possible_actions.append(
                        get_cards_array(card_type_in_hand, num_cards_in_board)
                    )

                # Or you can play dirty (with Joker(s))
                if card_type_in_hand != JOKER and hand[JOKER] > 0 \
                        and card_type_in_hand < card_type_in_board \
                        and num_cards_in_board >= 2 and hand[card_type_in_hand] > 0 \
                        and hand[card_type_in_hand] + hand[JOKER] >= num_cards_in_board:
                    # Use one joker
                    if hand[card_type_in_hand] + 1 >= num_cards_in_board:
                        possible_actions.append(
                            get_cards_array(card_type_in_hand,
                                            num_cards_in_board - 1)
                            + get_cards_array(JOKER, 1)
                        )

                    # Use two jokers
                    if hand[JOKER] == 2 and num_cards_in_board >= 3:
                        possible_actions.append(
                            get_cards_array(card_type_in_hand,
                                            num_cards_in_board - 2)
                            + get_cards_array(JOKER, 2)
                        )

    # If board empty, passing not allowed
    if np.all(board == 0):
        possible_actions = possible_actions[1:]

    return np.vstack(possible_actions)
