from typing import List, Optional

import numpy as np
import pandas as pd


class GameExtractor:
    """ Log game states.

    Extract structure:
    Game Index (1), Player Index (1), Ranking (1), Already played (13), Board (13), Hand (13) = 42
    """

    def __init__(self) -> None:
        self.game_states: Optional[np.ndarray] = None

    def record_state(
        self, game_index: int, player_index: int,
        already_played: np.ndarray, board: np.ndarray, hand: np.ndarray
    ) -> None:
        new_record = np.hstack([
            game_index, player_index, -1, already_played, board, hand
        ])

        if self.game_states is not None:
            self.game_states = np.vstack([self.game_states, new_record])
        else:
            self.game_states = new_record

    def set_game_ranking(self, game_index: int, ranking: List) -> None:
        if self.game_states is not None:
            for rank, p in enumerate(ranking):
                self.game_states[
                    (self.game_states[:, 0] == game_index) &
                    (self.game_states[:, 1] == p),
                    2
                ] = rank

    def export_log_to_file(self, file_path: str = "./egd/logs/egd_games.csv") -> None:
        game_df = pd.DataFrame(self.game_states, columns=[
            "gid", "pid", "rank",
            "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8", "ap9", "ap10", "ap11", "ap12", "ap13",
            "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11", "b12", "b13",
            "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11", "h12", "h13",
        ])
        game_df.to_csv(file_path, index=None)
