import numpy as np


class ttt_environment:
    state = np.zeros((9, 1))

    def make_move(self, player, position):
        if self.state[position] != 0:
            valid_move = 0
            terminal = 0
            reward = 0
            return valid_move, terminal, reward
        else:
            valid_move = 1

        self.state[position] = player

        terminal, reward = self.reward()

        return valid_move, terminal, reward

    def reward(self):

        reward = 0
        terminal = 0

        row_winner = max([self.state[0:3].sum(), self.state[3:6].sum(), self.state[6:9].sum()], key=abs)
        col_winner = max([self.state[0:7:3].sum(), self.state[1:8:3].sum(), self.state[2:9:3].sum()], key=abs)
        diag_winner = max(np.array([self.state[0::4].sum(), self.state[2::2].sum()]), key=abs)
        winner = max([row_winner, col_winner, diag_winner], key=abs)

        if abs(winner) == 3:
            terminal = 1
            reward = 1
        elif np.sum(self.state == 0) == 0:
            terminal = 1
            reward = -0.5

        return terminal, reward

    def reset(self):
        self.state = np.zeros((9, 1))
