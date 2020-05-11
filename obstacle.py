
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

        self.obstacles = self.assign_obstacles()

        # only consider terminal states with no obstacles as a goal
        self.goal_states = self.terminal_states.difference(self.obstacles)

        # define rewards and transitions
        self.trans_mat = self.build_trans_mat()
        self.reward_mat = self.build_reward_mat()

    def step(self, action: int) -> (int, int, bool):

        assert action in range(self._nA), "Invalid Action"

        prev_state = self.state
        self.state = self.trans_mat[self.state, action]
        r = self.reward_mat[prev_state, action]

        if self.state in self.goal_states:
            return self.state, r, True
        else:
            return self.state, r, False

    def build_trans_mat(self) -> np.ndarray:
        trans_mat = np.zeros((self._nS, self._nA), dtype=int)

        for s in range(self._nS):

            # cannot move once in terminal state
            if s in self.goal_states:
                trans_mat[s, :] = s
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
            if (s + 1) % self.nCols == 0:
                trans_mat[s][2] = s
            else:
                trans_mat[s][2] = s + 1

            # define down movements
            if s >= self._nS - self.nCols:
                trans_mat[s][3] = s
            else:
                trans_mat[s][3] = s + self.nCols

        return trans_mat


    def build_reward_mat(self) -> np.ndarray:

        reward_mat = np.zeros((self._nS, self._nA))

        for s in range(self._nS):

            # current row of state
            row = s // self.nCols

            # reached goal, reward for moving into goal
            if s in self.goal_states:
                reward_mat[s - 1, 2] = 50.

                # make sure it's not top or bottom rows of grid
                if row != self.nRows - 1:
                    reward_mat[s + self.nCols, 1] = 50.

                if row != 0:
                    reward_mat[s - self.nCols, 3] = 50.
                continue

            # hit obstacle, penalize for moving into state
            if s in self.obstacles:
                reward_mat[s - 1, 2] = -5.

                # obstacle in last column
                if s not in self.terminal_states:
                    reward_mat[s + 1, 0] = -15.

                if row != self.nRows - 1:
                    reward_mat[s + self.nCols, 1] = -5.

                if row != 0:
                    reward_mat[s - self.nCols, 3] = -5.

            # right movements
            reward_mat[s, 2] = 5.

            # left movements
            reward_mat[s, 0] = -10.

        return reward_mat



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








