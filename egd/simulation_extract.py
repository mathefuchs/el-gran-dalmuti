from typing import List

import numpy as np
import pandas as pd


class GameExtractor:
    """ Log game states.

    Extract structure:
    Game Index (1), Player Index (1), Ranking (1), Already played (13), Board (13), Hand (13) = 42
    """

    def __init__(self) -> None:
        self.game_states: List[np.ndarray] = []
        self.last_insert_idx: int = 0

    def record_state(
        self, game_index: int, player_index: int,
        already_played: np.ndarray, board: np.ndarray, hand: np.ndarray
    ) -> None:
        state = [game_index, player_index, -1]
        state.extend(already_played)
        state.extend(board)
        state.extend(hand)
        self.game_states.append(state)

    def set_game_ranking(self, ranking: List) -> None:
        ranking_dict = {p: rank for rank, p in enumerate(ranking)}

        for record in self.game_states[self.last_insert_idx:]:
            record[2] = ranking_dict[record[1]]

        self.last_insert_idx = len(self.game_states)

    def export_log_to_file(
        self, game_index: int, file_path: str = "./egd/logs/egd_games"
    ) -> None:
        game_df = pd.DataFrame.from_records(self.game_states, columns=[
            "gid", "pid", "rank",
            "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8", "ap9", "ap10", "ap11", "ap12", "ap13",
            "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11", "b12", "b13",
            "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11", "h12", "h13",
        ])
        idx = (game_index + 1) // 10000
        game_df.to_parquet(file_path + "_" + str(idx) + ".parquet", index=None)

        self.game_states: List[np.ndarray] = []
        self.last_insert_idx: int = 0
