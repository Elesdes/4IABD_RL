class LineWorld:
    def __init__(self):
        self.S = [0, 1, 2, 3, 4, 5, 6, 7]
        self.A = [0, 1]  # Gauche, Droite
        self.R = [-1.0, 0.0, 1.0]

    def p(self, s, a, s_p, r):
        assert (s >= 0 and s <= 7)
        assert (s_p >= 0 and s_p <= 7)
        assert (a >= 0 and a <= 1)
        assert (r >= 0 and r <= 2)
        if s == 0 or s == 4:
            return 0.0
        if s + 1 == s_p and a == 1 and r == 1 and s != 3:
            return 1.0
        if s + 1 == s_p and a == 1 and r == 2 and s == 3:
            return 1.0
        if s - 1 == s_p and a == 0 and r == 1 and s != 1:
            return 1.0
        if s - 1 == s_p and a == 0 and r == 0 and s == 1:
            return 1.0
        return 0.0

    def pi(self, s, a):
        if s == 0 or s == 4:
            return 0.0
        return 0.5
