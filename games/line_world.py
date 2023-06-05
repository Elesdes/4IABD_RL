from typing import List
class SingleAgentLineWorldEnv:
    def state_id(self) -> int:
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

class LineWorldEnv(SingleAgentLineWorldEnv):
    def __init__(self, cells_count: int):
        self.cells_count = cells_count
        self.agent_pos = -1
        self.reset()

    def state_id(self) -> int:
        return self.agent_pos

    def state_space(self) -> int:
        return self.cells_count

    def action_space(self) -> int:
        return 2

    def is_game_over(self) -> bool:
        return self.agent_pos == 0 or self.agent_pos == self.cells_count - 1

    def available_actions(self) -> List[int]:
        return [0, 1] if not self.is_game_over() else []  # Left, Right

    def step(self, action: int):
        assert (not self.is_game_over())
        assert (action in self.available_actions())
        self.agent_pos += -1 if action == 0 else 1

    def score(self) -> float:
        if self.agent_pos == 0:
            return -1.0
        if self.agent_pos == self.cells_count - 1:
            return 1.0
        return 0.0

    def possible_score(self) -> List:
        return [-1, 0, 1]
    def reset(self):
        self.agent_pos = self.cells_count // 2

    def view(self):
        for cell in range(self.cells_count):
            print('X' if cell == self.agent_pos else '_', end='')
        print()

    def pi(self, s, a):
        if s == 0 or s == self.cells_count-1:
            return 0.0
        return 0.5

    def p(self, s, a, s_p, r):
        assert (s >= 0 and s <= self.cells_count)
        assert (s_p >= 0 and s_p <= self.cells_count)
        assert (a >= 0 and a <= 1)
        assert (r >= 0 and r <= 2)
        if s == 0 or s == self.cells_count-1:
            return 0.0
        if s + 1 == s_p and a == 1 and r == 1 and s != self.cells_count-2:
            return 1.0
        if s + 1 == s_p and a == 1 and r == 2 and s == self.cells_count-2:
            return 1.0
        if s - 1 == s_p and a == 0 and r == 1 and s != 1:
            return 1.0
        if s - 1 == s_p and a == 0 and r == 0 and s == 1:
            return 1.0
        return 0.0