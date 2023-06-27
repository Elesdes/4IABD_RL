from typing import List


class SingleAgentGridWorldEnv:
    def state_id(self) -> int:
        pass

    def set_state_id(self, pos) -> None:
        pass

    def state_space(self) -> int:
        pass

    def action_space(self) -> int:
        pass

    def is_game_over(self) -> bool:
        pass

    def available_actions(self) -> List[int]:
        pass

    def step(self, action: int):
        pass

    def score(self) -> float:
        pass

    def possible_score(self) -> List:
        pass

    def reset(self):
        pass

    def view(self):
        pass

    def pi(self, s, a):
        pass

    def p(self, s, a, s_p, r):
        pass


class GridWorldEnv(SingleAgentGridWorldEnv):
    def __init__(self, row_len: int, column_len: int):
        row_len = max(row_len, 2)
        column_len = max(column_len, 2)
        self.row_len = row_len
        self.column_len = column_len
        self.agent_pos = -1
        self.reset()
        """
        self.S = [0, 1, 2, 3, 4,
                        5, 6, 7, 8, 9,
                        10, 11, 12, 13, 14,
                        15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24]
        self.A = [0, 1, 2, 3]  # Gauche, Droite, Bas, Haut
        self.R = [-1.0, 0.0, 1.0]
        """

    def state_id(self) -> int:
        return self.agent_pos

    def set_state_id(self, pos) -> None:
        self.agent_pos = pos

    def state_space(self) -> int:
        return self.row_len * self.column_len

    def action_space(self) -> int:
        return 4

    def is_game_over(self) -> bool:
        return self.agent_pos in [
            self.row_len - 1,
            (self.row_len * self.column_len) - 1,
        ]

    def available_actions(self) -> List[int]:
        if self.is_game_over():
            return []
        aa = [0, 1, 2, 3]
        if self.state_id() in list(
            range(0, self.row_len * self.column_len - 1, self.row_len)
        ):
            aa.remove(0)
        if self.state_id() in list(
            range(self.row_len - 1, self.row_len * self.column_len, self.row_len)
        ):
            aa.remove(1)
        if self.state_id() in list(
            range(
                self.row_len * self.column_len - self.row_len,
                self.row_len * self.column_len,
            )
        ):
            aa.remove(2)
        if self.state_id() in list(range(self.row_len)):
            aa.remove(3)
        return aa

    def step(self, action: int):
        assert not self.is_game_over()
        assert action in self.available_actions()
        # [0, 1, 2, 3] Gauche, Droite, Bas, Haut
        if action == 0:
            self.agent_pos += -1
        elif action == 1:
            self.agent_pos += 1
        elif action == 2:
            self.agent_pos += self.row_len
        elif action == 3:
            self.agent_pos -= self.row_len

    def score(self) -> float:
        if self.agent_pos == self.row_len - 1:
            return -1.0
        return 1.0 if self.agent_pos == ((self.row_len * self.column_len) - 1) else 0.0

    def possible_score(self) -> List:
        return [-1, 0, 1]

    def reset(self):
        self.agent_pos = 0

    def view(self):
        for cell in range(self.row_len * self.column_len):
            print("X" if cell == self.agent_pos else "_", end="")
        print()

    def p(self, s, a, s_p, r) -> float:
        if s == self.row_len - 2 and a == 1 and s_p == self.row_len - 1 and r == 0:
            return 1.0
        if s == 2 * self.row_len - 1 and a == 3 and s_p == self.row_len - 1 and r == 0:
            return 1.0
        if (
            s == (self.row_len * self.column_len) - 2
            and a == 1
            and s_p == (self.row_len * self.column_len) - 1
            and r == 2
        ):
            return 1.0
        if (
            s == ((self.row_len * self.column_len) - self.row_len) - 1
            and a == 2
            and s_p == (self.row_len * self.column_len) - 1
            and r == 2
        ):
            return 1.0

        already_stated_cases = [self.row_len - 1, self.row_len * self.column_len - 1]
        left_column = list(range(0, self.row_len * self.column_len - 1, self.row_len))
        state_cases = [
            x
            for x in range(self.row_len * self.column_len)
            if x not in already_stated_cases + left_column
        ]
        if s in state_cases and a == 0 and s_p == s - 1 and r == 1:
            return 1.0

        already_stated_cases = [self.row_len - 2, self.row_len * self.column_len - 2]
        right_column = list(
            range(self.row_len - 1, self.row_len * self.column_len, self.row_len)
        )
        state_cases = [
            x
            for x in range(self.row_len * self.column_len)
            if x not in already_stated_cases + right_column
        ]
        if s in state_cases and a == 1 and s_p == s + 1 and r == 1:
            return 1.0

        already_stated_cases = [
            self.row_len * self.column_len - self.row_len - 1,
            self.row_len - 1,
        ]
        bottom_row = list(
            range(
                self.row_len * self.column_len - self.row_len,
                self.row_len * self.column_len,
            )
        )
        state_cases = [
            x
            for x in range(self.row_len * self.column_len)
            if x not in already_stated_cases + bottom_row
        ]
        if s in state_cases and a == 2 and s_p == s + self.row_len and r == 1:
            return 1.0

        already_stated_cases = [
            2 * self.row_len - 1,
            self.row_len * self.column_len - 1,
        ]
        top_row = list(range(self.row_len))
        state_cases = [
            x
            for x in range(self.row_len * self.column_len)
            if x not in already_stated_cases + top_row
        ]
        if s in state_cases and a == 3 and s_p == s - self.row_len and r == 1:
            return 1.0
        return 0.0

    def pi(self, s, a):
        if s in [self.row_len - 1, self.row_len * self.column_len - 1]:
            return 0.0
        return 0.25
