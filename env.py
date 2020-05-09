
class GridWorld(object):
    """
    Grid world environment
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
        self.terminal_states = [i for i in self.terminal_states()]


    def terminal_states(self):
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
        return NotImplementedError()

    @property
    def nA(self):
        """ # of possible actions """
        return NotImplementedError()

    @property
    def gamma(self):
        """ discount factor """
        return NotImplementedError()

    @property
    def TD(self):
        """
        Transition Dynamics
        return: a numpy array shape of [nS,nA,nS]
            TD[s,a,s'] := the probability it will resulted in s' when it execute action a given state s
        """
        return NotImplementedError()

    @property
    def R(self):
        """
        Reward function
        return: a numpy array shape of [nS,nA,nS]
            R[s,a,s'] := reward the agent will get it experiences (s,a,s') transition.
        """
        return NotImplementedError()

    def sample(self):
        """
        :return: Random action for exploration
        """
        return NotImplementedError()

    def reset(self):
        """
        Reset the environment when you want to generate a new episode.
        Randomly initializes location at beginning of sidewalk.
        return:
            initial state
        """
        return NotImplementedError()

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
