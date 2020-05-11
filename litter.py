from typing import Set
import numpy as np
from env import GridWorld

class LitterEnv(GridWorld):
    """
        Responsible for randomly placing litter in the gridworld and
        defining the reward matrix for each action.
        :author Isaac Buitrago
    """

    def __init__(self, nRows, nCols, nA, gamma):
        super(LitterEnv, self).__init__(nRows, nCols, nA, gamma)

        self.litter = self.assign_litter()

        # define rewards and transitions
        self.trans_mat = self.build_trans_mat()
        self.reward_mat = self.build_reward_mat()

    def step(self, action: int) -> (int, int, bool):

        prev_state = self.state
        self.state = self.trans_mat[self.state, action]
        r = self.reward_mat[prev_state, action]

        if self.state in self.terminal_states:
            return self.state, r, True
        else:
            return self.state, r, False

    def build_trans_mat(self) -> np.ndarray:

        trans_mat = np.zeros((self.nS, self.nA), dtype=int)

        for s in range(self.nS):

            # cannot move once in terminal state
            if s in self.terminal_states:
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
            if s >= self.nS - self.nCols:
                trans_mat[s][3] = s
            else:
                trans_mat[s][3] = s + self.nCols

        return trans_mat


    def build_reward_mat(self) -> np.ndarray:
        reward_mat = np.zeros((self.nS, self.nA))

        for s in range(self.nS):

            # current row of state
            row = s // self.nCols

            # reached goal, reward for moving into goal
            if s in self.terminal_states:
                reward_mat[s - 1, 2] = 50.
                continue

            # reward for picking up litter
            if s in self.litter:
                reward_mat[s - 1, 2] = 5.

                # litter not in the last column
                # if s not in self.terminal_states:
                #     reward_mat[s + 1, 0] = 5.

                if row != 0:
                    reward_mat[s - self.nCols, 3] = 5.

                if row != self.nRows - 1:
                    reward_mat[s + self.nCols, 1] = 5.


            # right movements
            reward_mat[s, 2] = 10.

            # left movements
            reward_mat[s, 0] = -10.

        return reward_mat


    def assign_litter(self) -> Set[int]:
        """
        Randomly places litter throughout gridworld,
        where every cell has a 30% of being selected.
        :return: set of states with litter
        """

        dist = [1, 0, 1, 0, 0, 0, 1, 0, 0, 0]

        litter = set()

        for s in range(self.nS):

            sample = np.random.choice(dist)

            if sample == 1:
                litter.add(s)

        return litter
