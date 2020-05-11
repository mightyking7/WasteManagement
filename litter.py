from env import GridWorld

class LitterEnv(GridWorld):

    def __init__(self, nRows, nCols, nA, gamma):
        super().__init__(nRows, nCols, nA, gamma)

    def reset(self):
        return super().reset()

    def step(self, action: int):
        return super().step(action)

    def build_trans_mat(self):
        return super().build_trans_mat()

    def build_reward_mat(self):
        return super().build_reward_mat()