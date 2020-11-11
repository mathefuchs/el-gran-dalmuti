import numpy as np

from egd.game.cards import NUM_CARD_VALUES, JOKER
from egd.game.state import has_finished, get_cards_array


def possible_next_moves(hand, board):
    """
        Returns possible next moves as a list of tuples (new hand, new board)
    """

    card_type_in_board = np.argmax(board)
    num_cards_in_board = board[card_type_in_board] + board[JOKER]

    # You can always pass if it is not the initial move
    possible_hands = np.reshape(hand, (1, NUM_CARD_VALUES))
    possible_boards = np.reshape(board, (1, NUM_CARD_VALUES))

    if not has_finished(hand):
        for card_type_in_hand in range(NUM_CARD_VALUES - 1, -1, -1):
            # You can play clean
            if card_type_in_hand < card_type_in_board and \
                    hand[card_type_in_hand] >= num_cards_in_board:
                new_board = get_cards_array(
                    card_type_in_hand, num_cards_in_board)
                new_hand = hand - new_board
                possible_hands = np.vstack([possible_hands, new_hand])
                possible_boards = np.vstack([possible_boards, new_board])

            # Or you can play dirty (with Joker(s))
            if card_type_in_hand != JOKER and hand[JOKER] > 0 and \
                    card_type_in_hand < card_type_in_board and hand[card_type_in_hand] > 0 and \
                    hand[card_type_in_hand] + hand[JOKER] >= num_cards_in_board:
                # Use one joker
                if hand[card_type_in_hand] + 1 >= num_cards_in_board:
                    joker_vec = get_cards_array(JOKER, 1)
                    new_board = get_cards_array(
                        card_type_in_hand, num_cards_in_board - 1) + joker_vec
                    new_hand = hand - new_board
                    possible_hands = np.vstack([possible_hands, new_hand])
                    possible_boards = np.vstack([possible_boards, new_board])

                # Use two jokers
                if hand[JOKER] == 2 and num_cards_in_board > 2:
                    joker_vec = get_cards_array(JOKER, 2)
                    new_board = get_cards_array(
                        card_type_in_hand, num_cards_in_board - 2) + joker_vec
                    new_hand = hand - new_board
                    possible_hands = np.vstack([possible_hands, new_hand])
                    possible_boards = np.vstack([possible_boards, new_board])

    return possible_hands, possible_boards
