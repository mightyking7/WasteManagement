import numpy as np
from env import GridWorld

class ForwardEnv(GridWorld):
    """
        Defines the states, actions, and rewards an agent
        needs to learn an optimal policy for moving to the opposite
        end of the board as fast as possible.

        :author Isaac Buitrago
    """
    def __init__(self, nRows, nCols, nA, gamma):
        super(ForwardEnv, self).__init__(nRows, nCols, nA, gamma)
        self.trans_mat = self.build_trans_mat()
        self.reward_mat = self.build_reward_mat()


    def reset(self) -> int:
        """
        Reset the environment when you want to generate a new episode.
        Randomly initializes location in first column of the board.
        return:
            initial state
        """
        # states in first column
        first_col = np.array([i - self.nCols + 1 for i in self.terminal_states])
        self.state = np.random.choice(first_col)
        return self.state

    def step(self, action:int) -> (int, int, bool):
        """
        proceed one step in the state space.
        return:
            next state, reward, done (whether agent reached a terminal state)
        """
        assert action in range(self._nA), "Invalid Action"

        prev_state = self.state
        self.state = self.trans_mat[self.state, action]
        r = self.reward_mat[prev_state, action]

        if self.state in self.terminal_states:
            return self.state, r, True
        else:
            return self.state, r, False


    def build_trans_mat(self) -> np.ndarray:
        """
        Defines the transition dynamics matrix.
        :return:
        """
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
            trans_mat[s][2] = s + 1

            # define down movements
            if s >= self._nS - self.nCols:
                trans_mat[s][3] = s
            else:
                trans_mat[s][3] = s + self.nCols

        return trans_mat

    def build_reward_mat(self) -> np.ndarray:
        """
        Defines the reward matrix
        :return:
        """
        reward_mat = np.zeros((self.nS, self.nA))

        for s in range(self.nS):

            row = s // self.nCols

            # terminal state
            if s in self.terminal_states:
                reward_mat[s - 1, 2] = 50.
                continue

            # right movement
            reward_mat[s, 2] = 10.

            # left movement
            reward_mat[s, 0] = -10.

        return reward_mat