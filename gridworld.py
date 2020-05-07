import numpy as np
from env import GridWorld
# from algo import Q_learning
from tqdm import tqdm

# rows in grid
rows = 6

# columns in grid
cols = 25

# probability of taking a greedy action
epsilon = 0.9

# discount factor
gamma = 0.6

# num actions: up, down, left, right
nA = 4

# learning rate
lr = 0.01

# episodes for training
n_episodes = 50


# 1 is an obstacle, 0 is an empty cell
# dist = [1,0,0,0,0]
#
# num_tiles = rows * cols
#
# world = np.array([np.random.choice(dist) for i in range (num_tiles)]).reshape((rows, cols))

grid_world = GridWorld(nRows = rows, nCols = cols, nA = nA, gamma = gamma)
# behavior_policy = RandomPolicy()


# initial Q table
Q = np.zeros((grid_world.nS, grid_world.nA))


for _ in tqdm(range(n_episodes)):

    state = grid_world.reset()

    done = False

    while not done :

        # explore action space
        if np.random.uniform(0, 1) < epsilon:
            action = grid_world.sample()

        # exploit
        else:
            action = np.argmax(Q[state])

        # take step in env
        obs, r, done = grid_world.step(action)

        # update Q table
        prev_value = Q[state, action]

        new_value = prev_value + lr * (r + grid_world.gamma * np.max(Q[obs]) - prev_value)

        Q[state, action] = new_value

        # update state
        state = obs

# sidewalk policy
# Q = Q_learning(grid_world.gamma, trajs, behavior_policy, lr = lr,
#                initQ=np.zeros((grid_world.nS, grid_world.nA)))

# plot policy


