import os
from algo import Q_learning
import numpy as np
from env import GridWorld

class Trainer:

    def __init__(self, env: GridWorld):

        self.env = env
        self.policy_dir = "./policy/"

    def train(self, fname, epsilon, lr, n_episodes) -> np.ndarray:

        fname_policy = self.policy_dir + fname

        # train if policy doesn't exist
        if not os.path.exists(fname_policy):

            # randomly initialize Q table
            Q = np.random.rand(self.env.nS, self.env.nA)

            # set state-action value of final states to 0
            # TODO make this dynamic
            Q[[24, 49, 74, 99, 124, 149], :] = 0

            for e in range(n_episodes):
                Q, _ = Q_learning(env=self.env, epsilon=epsilon, lr=lr, initQ=Q)
                print(f"> Episode: {e}")

            # save policy
            np.save(fname_policy, Q)

        else:
            # load policy
            Q = np.load(fname_policy)

        return Q