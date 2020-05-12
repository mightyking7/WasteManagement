import numpy as np
from env import GridWorld

class Sidewalk(GridWorld):
    """
        Defines the states, actions, and rewards an agent
        needs to learn an optimal policy for staying on the sidewalk.

        :author Isaac Buitrago
    """
    def __init__(self, nRows, nCols, sL, sR,  nA, gamma):
        super(Sidewalk, self).__init__(nRows, nCols, nA, gamma)

        self.sL = sL
        self.sR = sR
        self.trans_mat = self.build_trans_mat()
        self.reward_mat = self.build_reward_mat()


    def reset(self) -> int:
        """
        Reset the environment when you want to generate a new episode.
        Randomly initializes location at beginning of sidewalk.
        return:
            initial state
        """
        first_col = np.array([i - self.nCols + 1 for i in self.terminal_states])

        # states at beginning of sidewalk
        ss = list()
        for s in first_col:
            row = s // self.nCols
            if row > self.sL and row < self.sR:
                ss.append(row)

        self.state = self.nCols * np.random.choice(ss)

        return self.state

    def step(self, action:int) -> (int, int):
        """
        proceed one step in the state space.
        return:
            next state and reward
        """
        assert action in range(self._nA), "Invalid Action"

        prev_state = self.state
        self.state = self.trans_mat[self.state, action]
        r = self.reward_mat[prev_state, action]
        return self.state, r


    def build_trans_mat(self) -> np.ndarray:
        """
        Defines the transition dynamics matrix.
        :return:
        """
        trans_mat = np.zeros((self.nS, self.nA), dtype=int)

        for s in range(self.nS):

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
            if s in self.terminal_states:
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
        """
        Defines the reward matrix for
        transitions in the environment.
        Initially, all transitions are 10 and penalties
        are given for touching the left and right sidewalk edges.
        :return:
        """
        reward_mat = np.zeros((self.nS, self.nA))

        for s in range(self.nS):

            row = s // self.nCols

            # on left edge of sidewalk
            if row == self.sL:
                # penalize movement into and along edge
                reward_mat[s + self.nCols, 1] = -10.
                reward_mat[s, 0] = -10.
                reward_mat[s, 2] = -10.

                # reward movement out of left edge
                reward_mat[s, 3] = 10.
                continue

            # on right edge of sidewalk
            elif row == self.sR:
                # penalize movement into and along edge
                reward_mat[s - self.nCols, 3] = -10.
                reward_mat[s, 0] = -10.
                reward_mat[s, 2] = -10.

                # reward movement out of right edge
                reward_mat[s, 1] = 10.
                continue

            # reward rightward movements
            if s not in self.terminal_states:
                reward_mat[s, 2] = 20.

            # define leftward movements
            if s % self.nCols != 0:
                reward_mat[s, 0] = -10.

            # define downward movements
            # if s < self.nS - self.nCols:
            #     reward_mat[s, 3] = 10.


        return reward_mat