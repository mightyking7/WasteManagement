from typing import Set
import numpy as np
from env import GridWorld

class LitterEnv(GridWorld):

    def __init__(self, nRows, nCols, nA, gamma):
        super(LitterEnv, self).__init__(nRows, nCols, nA, gamma)

        self.litter = self.assign_litter()

    def step(self, action: int):
        return super().step(action)

    def build_trans_mat(self):
        return super().build_trans_mat()

    def build_reward_mat(self):
        return super().build_reward_mat()

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
