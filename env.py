import numpy as np

class GridWorld(object):
    """
    Grid world environment with a set number of rows, columns, and states.
    The environment also defines the number of actions and the discount factor.
    Child classes are responsible for defining the reward matrix, transition matrix,
    and how to step through the environment.

    :author: Isaac Buitrago
    """
    def __init__(self, nRows, nCols, nA, gamma):
        """
        :param nRows: number of rows in grid
        :param nCols: number of cols in grid
        :param nA: Number of actions
        :param gamma: discount factor
        """
        self._nS = nRows * nCols
        self._nA = nA
        self._gamma = gamma
        self.nRows = nRows
        self.nCols = nCols

        """
        0 - left
        1 - up
        2 - right
        3 - down
        """
        self.action_space = [0, 1, 2, 3]
        self.terminal_states = {i for i in self.get_terminal_states()}


    def get_terminal_states(self):
        """
        Generator yields terminal states
        :return: terminal state in grid world
        """
        for i in range(1, self._nS + 1):
           if i % self.nCols == 0:
               yield i - 1

    @property
    def nS(self):
        """ # of possible states """
        return self._nS

    @property
    def nA(self):
        """ # of possible actions """
        return self._nA

    @property
    def gamma(self):
        """ discount factor """
        return self._gamma

    def sample(self) -> int:
        """
        :return: Random action for exploration
        """
        return np.random.choice(self.action_space)


    def reset(self) -> int:
        """
        Reset the environment when you want to generate a new episode.
        Randomly initializes location at beginning of grid.
        return:
            initial state
        """
        # states in first column
        first_col = np.array([i - self.nCols + 1 for i in self.terminal_states])
        self.state = np.random.choice(first_col)
        return self.state

    def step(self, action:int):
        """
        proceed one step in the state space.
        return:
            next state, reward, done (whether agent reached to a terminal state)
        """
        return NotImplementedError()

    def build_trans_mat(self):
        """
        Defines the transition dynamics matrix.
        :return:
        """
        return NotImplementedError()

    def build_reward_mat(self):
        """
        Defines the reward matrix
        :return:
        """
        return NotImplementedError()
