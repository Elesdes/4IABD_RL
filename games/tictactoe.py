import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1

    def play(self, action):
        if self.board[action] == 0:
            self.board[action] = self.player
            self.player = -self.player
            return True
        return False

    def is_game_over(self):
        # Check rows
        for row in self.board:
            if sum(row) == 3 or sum(row) == -3:
                return True

        # Check columns
        for col in self.board.T:
            if sum(col) == 3 or sum(col) == -3:
                return True

        # Check diagonals
        if self.board[0][0] + self.board[1][1] + self.board[2][2] == 3 \
                or self.board[0][0] + self.board[1][1] + self.board[2][2] == -3:
            return True
        elif self.board[0][2] + self.board[1][1] + self.board[2][0] == 3 \
                or self.board[0][2] + self.board[1][1] + self.board[2][0] == -3:
            return True

        return False

    def available_actions(self):
        return np.argwhere(self.board == 0)

    def print_board(self):
        print(self.board)


if __name__ == "__main__":
    game = TicTacToe()
    game.print_board()
    game.reset()
    game.print_board()
