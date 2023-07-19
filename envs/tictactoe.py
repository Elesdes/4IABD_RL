import numpy as np

from do_not_touch.contracts import SingleAgentEnv


class TicTacToe(SingleAgentEnv):
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1

    def reset_random(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1

    def is_game_over(self):
        return self.check_winner() or self.is_board_full()

    def state_id(self):
        return str(self.board)

    def available_actions_ids(self):
        return [i for i, val in enumerate(self.board) if val == 0]

    def act_with_action_id(self, action_id):
        self.board[action_id] = self.current_player
        self.current_player = -self.current_player

    def score(self):
        winner = self.check_winner()
        if winner == 1:
            return 1
        elif winner == -1:
            return -1
        else:
            return 0

    def check_winner(self):
        winning_moves = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]

        return next(
            (
                self.board[moves[0]]
                for moves in winning_moves
                if self.board[moves[0]]
                   == self.board[moves[1]]
                   == self.board[moves[2]]
                   != 0
            ),
            None,
        )

    def is_board_full(self):
        return np.all(self.board != 0)
