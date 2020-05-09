import numpy as np
from env import GridWorld

class Sidewalk(GridWorld):
    """
    Defines the states, actions, and rewards the
    agent needs to learn an optimal policy.
    """
    def __init__(self, nRows, nCols, sL, sR,  nA, gamma):
        super(GridWorld, self).__init__(nRows, nCols, nA, gamma)

        self._nS = nRows * nCols
        self._nA = nA
        self._gamma = gamma
        self.nRows = nRows
        self.nCols = nCols
        self.sL = sL
        self.sR = sR

        # left, up, right, down
        self.action_space = [0, 1, 2, 3]

        self.terminal_states = [i for i in self.terminal_states()]
        self._trans_mat = self.build_trans_mat()
        self._reward_mat = self.build_reward_mat()


    @property
    def nS(self) -> int:
        """ # of possible states """
        return self._nS

    @property
    def nA(self) -> int:
        """ # of possible actions """
        return self._nA

    @property
    def gamma(self) -> float:
        """ discount factor """
        return self._gamma

    @property
    def TD(self) -> np.array:
        """
        Transition Dynamics
        return: a numpy array shape of [nS,nA,nS]
            TD[s,a,s'] := the probability it will resulted in s' when it execute action a given state s
        """
        return self._trans_mat

    @property
    def R(self) -> np.array:
        """
        Reward function
        return: a numpy array shape of [nS,nA,nS]
            R[s,a,s'] := reward the agent will get it experiences (s,a,s') transition.
        """
        return self._reward_mat

    def sample(self):
        """
        :return: Random action for exploration
        """
        return np.random.choice(self.action_space)

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

    def step(self, action:int) -> (int, int, bool):
        """
        proceed one step in the state space.
        return:
            next state, reward, done (whether agent reached to a terminal state)
        """
        assert action in range(self._nA), "Invalid Action"
        # assert self.state not in self.terminal_state, "Episode has ended!"

        prev_state = self.state
        self.state = np.random.choice(self.nS, p=self._trans_mat[self.state, action])
        r = self._reward_mat[prev_state, action, self.state]

        if self.state in self.terminal_states:
            return self.state, r, True
        else:
            return self.state, r, False


    def build_trans_mat(self) -> np.ndarray:
        """
        Defines the transition dynamics matrix.
        :return:
        """
        trans_mat = np.zeros((self._nS, self._nA, self._nS), dtype=int)

        for s in range(self._nS):

            # cannot move once in terminal state
            if s in self.terminal_states:
                trans_mat[s, :, s] = 1.
                continue

            # define left movements
            if s % self.nCols == 0:
                trans_mat[s][0][s] = 1.
            else:
                trans_mat[s][0][s - 1] = 1.

            # define up movements
            if s < self.nCols:
                trans_mat[s][1][s] = 1.
            else:
                trans_mat[s][1][s - self.nCols] = 1.

            # define right movements
            trans_mat[s][2][s + 1] = 1.

            # define down movements
            if s >= self._nS - self.nCols:
                trans_mat[s][3][s] = 1.
            else:
                trans_mat[s][3][s + self.nCols] = 1.

        return trans_mat

    def build_reward_mat(self) -> np.ndarray:
        """
        Defines the reward matrix
        :return:
        """
        reward_mat = np.zeros((self._nS, self._nA, self._nS))

        for s in range(self._nS):

            row = s // self.nCols

            # terminal state
            if s in self.terminal_states:

                # within sidewalk
                if row > self.sL and row < self.sR:
                    reward_mat[s - 1, 2, s] = 50.
                else:
                    reward_mat[s - 1, 2, s] = 15.

                continue

            # within sidewalk
            if row > self.sL and row < self.sR:
                reward_mat[s, 2, s + 1] = 10.
                reward_mat[s, 0, s - 1] = -10.

            # on left sidewalk edge
            elif row == self.sL:
                reward_mat[s, 3, s + self.nCols] = 10.
                reward_mat[s, 2, s + 1] = 5.
                reward_mat[s, 0, s - 1] = -10.

            # on right sidewalk edge
            elif row == self.sR:
                reward_mat[s, 1, s - self.nCols] = 10.
                reward_mat[s, 2, s + 1] = 5.
                reward_mat[s, 0, s - 1] = -10.

            # out of sidewalk
            else:
                reward_mat[s, 2, s + 1] = 5.
                reward_mat[s, 0, s - 1] = -10.

        return reward_mat