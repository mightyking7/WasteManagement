
from env import GridWorld
import numpy as np
from typing import Set

class ObstacleEnv(GridWorld):
    """
        Defines the states, actions, and rewards an
        agent needs to learn an optimal policy for avoiding obstacles.
        :author Isaac Buitrago
    """

    def __init__(self, nRows, nCols, nA, gamma):
        super(ObstacleEnv, self).__init__(nRows, nCols, nA, gamma)

        # define rewards and transitions
        self.trans_mat = self.build_trans_mat()
        self.reward_mat = self.build_reward_mat()
        self.obstacles = self.assign_obstacles()

    def sample(self):
        return super().sample()

    def reset(self) -> int:
        # states in first column
        first_col = np.array([i - self.nCols + 1 for i in self.terminal_states])
        self.state = np.random.choice(first_col)
        return self.state

    def step(self, action: int) -> (int, int, bool):

        assert action in range(self._nA), "Invalid Action"

        prev_state = self.state
        self.state = np.random.choice(self.nS, p=self.trans_mat[self.state, action])
        r = self.reward_mat[prev_state, action]

        if self.state in self.terminal_states:
            return self.state, r, True
        else:
            return self.state, r, False

    def build_trans_mat(self):
        trans_mat = np.zeros((self._nS, self._nA), dtype=int)

        for s in range(self._nS):

            # self.obstacles.intersection(set(self.terminal_states))

            # cannot move once in terminal state
            if s in self.terminal_states :
                trans_mat[s, :, s] = 1.
                continue

            # define left movements
            if s % self.nCols == 0:
                trans_mat[s][0] = s
            else:
                trans_mat[s][0] = s - 1

            # define up movements
            if s < self.nCols:
                trans_mat[s][1] = s
            else:
                trans_mat[s][1] = s - self.nCols

            # define right movements
            trans_mat[s][2] = s + 1

            # define down movements
            if s >= self._nS - self.nCols:
                trans_mat[s][3] = s
            else:
                trans_mat[s][3] = s + self.nCols

        return trans_mat


    def build_reward_mat(self):
        return super().build_reward_mat()

    def assign_obstacles(self) -> Set[int]:
        """
        Randomly places obstacles in the gridworld, where each state
        has a 1/5 chance of being selected.
        :return: set of states with obstacles
        """

        dist = [1, 0, 0, 0, 0]

        obstacles = set()

        for s in range(self.nS):

            sample = np.random.choice(dist)

            if sample == 1:
                obstacles.add(s)

        return obstacles








