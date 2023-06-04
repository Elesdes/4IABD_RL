class GridWorld:
    def __init__(self):
        self.S = [0, 1, 2, 3, 4,
                        5, 6, 7, 8, 9,
                        10, 11, 12, 13, 14,
                        15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24]
        self.A = [0, 1, 2, 3]  # Gauche, Droite, Bas, Haut
        self.R = [-1.0, 0.0, 1.0]

    def p(self, s, a, s_p, r) -> float:
        if s == 3 and a == 1 and s_p == 4 and r == 0:
            return 1.0
        if s == 9 and a == 3 and s_p == 4 and r == 0:
            return 1.0
        if s == 23 and a == 1 and s_p == 24 and r == 2:
            return 1.0
        if s == 19 and a == 2 and s_p == 24 and r == 2:
            return 1.0
        if s in [1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22,
                 23] and a == 0 and s_p == s - 1 and r == 1:
            return 1.0
        if s in [0, 1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21,
                 22] and a == 1 and s_p == s + 1 and r == 1:
            return 1.0
        if s in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] and a == 2 and s_p == s + 5 and r == 1:
            return 1.0
        if s in [5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                 23] and a == 3 and s_p == s - 5 and r == 1:
            return 1.0
        return 0.0

    def pi(self, s, a):
        if s == 4 or s == 24:
            return 0.0
        return 0.25