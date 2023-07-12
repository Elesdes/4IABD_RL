import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1
        self.num_cells = 9
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1
        self.winner = None

    def play(self, action):
        row, col = action
        if self.board[row, col] == 0:
            self.board[row, col] = self.player
            if self.is_game_over():
                self.winner = self.player
            self.player = -self.player
            return True
        return False

    def is_game_over(self):
        # Check rows, columns, and diagonals 
        for row in self.board:
            if sum(row) in [3, -3]:
                return True
        return next(
            (True for col in self.board.T if sum(col) in [3, -3]),
            True
            if self.board[0][0] + self.board[1][1] + self.board[2][2] in [3, -3]
            else self.board[0][2] + self.board[1][1] + self.board[2][0] in [3, -3],
        )

    def available_actions(self):
        return np.argwhere(self.board == 0)
    
    def state_space(self):
        return self.num_cells

    def print_board(self):
        print(self.board)

    def get_winner(self):
        return self.winner


if __name__ == "__main__":
    game = TicTacToe()
    game.print_board()
    game.reset()
    game.print_board()
    